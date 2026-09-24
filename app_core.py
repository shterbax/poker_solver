import json
import time
import queue
import threading
import traceback
from typing import Optional
import mss
import numpy as np
import cv2
import pygetwindow as gw

from core.config import AppConfig
from core.models import TableState, GameType, PlayerState, PlayerStatus
from capture.frame_cropper import FrameCropper
from vision.card_detector import CardDetector
from vision.turn_detector import TurnDetector
from vision.stack_pot_detector import StackPotDetector
from vision.status_detector import StatusDetector
from vision.dealer_detector import DealerDetector
from vision.bet_detector import BetDetector
from vision.tournament_detector import TournamentInfoCollector
from core.action_tracker import ActionTracker
from llm.pokerbench import PokerBenchContextBuilder, LLMSolverWorker, EngineDecision


class AppCore:
    config: AppConfig

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()

        # Загрузка JSON-профиля для StackPotDetector
        self.profile_dict = self._load_profile_dict()

        # Инициализация всех компонент распознавания
        self.cropper = FrameCropper(str(self.config.profile_path))
        self.card_detector = CardDetector(self.config)
        self.turn_detector = TurnDetector(self.config)
        self.stack_pot_detector = StackPotDetector()
        self.status_detector = StatusDetector(self.config)
        self.dealer_detector = DealerDetector(threshold=self.config.dealer_detector_threshold)
        self.bet_detector = BetDetector()

        # Детектор турнирной информации (использует OCR из stack_pot_detector)
        self.tournament_detector = TournamentInfoCollector(ocr_engine=self.stack_pot_detector.ocr)

        # Трекер истории экшена на постфлопе
        self.action_tracker = ActionTracker()

        self._last_known_dealer_seat_id: Optional[int] = None

        _orig_recognize = self.stack_pot_detector.ocr.recognize
        self.stack_pot_detector.ocr.recognize = lambda crop: _orig_recognize(crop) or 0.0

        # Безопасное извлечение названия клиента
        client_val = getattr(self.config, "client", "coinpoker")
        if isinstance(client_val, dict):
            self.window_keyword = client_val.get("name") or client_val.get("title") or "coinpoker"
        else:
            self.window_keyword = str(client_val)

        # Потокобезопасные очереди
        self.state_queue = queue.Queue(maxsize=1)
        self.ui_queue = queue.Queue()

        self.solver_worker = LLMSolverWorker(
            api_url=self.config.llm_url,
            model_name=self.config.llm_model
        )

        self._is_running = False
        self._vision_thread: Optional[threading.Thread] = None
        self._solver_thread: Optional[threading.Thread] = None

        self._last_processed_hand_id: Optional[str] = None
        self._mode = "SMART"
        self.hero_seat_id = 0

    def _load_profile_dict(self) -> dict:
        """Вспомогательный метод для загрузки словаря профиля разметки стола."""
        try:
            if hasattr(self.config, "profile_data") and self.config.profile_data:
                return self.config.profile_data
            if hasattr(self.config, "profile_path") and self.config.profile_path.exists():
                with open(self.config.profile_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка загрузки JSON профиля для StackPotDetector: {e}")
        return {}

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_game_type(self, game_type: str) -> None:
        """Динамически переключает тип игры и подгружает соответствующий профиль разметки."""
        profile_map = {
            "CASH_6MAX": "coinpoker_6max_cash.json",
            "MTT_7MAX": "coinpoker_7max_mtt.json"
        }

        self.config.game_type = game_type

        if game_type in profile_map:
            self.config.active_profile = profile_map[game_type]

            # Перезагружаем словарь разметки и кроппер на лету
            self.profile_dict = self._load_profile_dict()
            self.cropper = FrameCropper(str(self.config.profile_path))
            print(f"🔄 Загружен новый профиль разметки: {self.config.active_profile}")

        # Сохраняем обновленный конфиг
        self.config.save()

    def set_cash_stakes(self, stakes: str) -> None:
        """Динамически обновляет лимит игры и сохраняет его в конфиг."""
        self.config.cash_stakes = stakes
        self.config.save()
        print(f"💰 Установлен новый лимит кэш-игры: {stakes}")

    def start(self) -> None:
        if self._is_running:
            return

        self._is_running = True
        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="VisionThread")
        self._solver_thread = threading.Thread(target=self._solver_loop, daemon=True, name="SolverThread")

        self._vision_thread.start()
        self._solver_thread.start()
        print("🚀 Ядро приложения и фоновые потоки успешно запущены.")

    def stop(self) -> None:
        self._is_running = False
        print("🛑 Запрошена остановка ядра.")

    def _get_poker_window_rect(self) -> Optional[dict]:
        target_title = self.window_keyword
        if isinstance(target_title, dict):
            target_title = target_title.get("name") or target_title.get("title") or "CoinPoker"

        windows = gw.getWindowsWithTitle(str(target_title))
        if not windows:
            for kw in ["CoinPoker", "Poker", "Hold'em", "Table"]:
                windows = gw.getWindowsWithTitle(kw)
                if windows:
                    break

        if windows:
            win = windows[0]
            if not win.isMinimized and win.width > 100 and win.height > 100:
                return {
                    "top": win.top,
                    "left": win.left,
                    "width": win.width,
                    "height": win.height
                }
        return None

    # =========================================================================
    # ПОТОК 1: COMPUTER VISION
    # =========================================================================
    def _vision_loop(self) -> None:
        saved_debug_img = False

        with mss.mss() as sct:
            while self._is_running:
                try:
                    window_rect = self._get_poker_window_rect()
                    monitor = window_rect if window_rect is not None else sct.monitors[1]

                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)[:, :, :3]

                    if not saved_debug_img and frame.size > 0:
                        cv2.imwrite("debug_poker_table.png", frame)
                        print(f"📸 Сохранен отладочный кадр: debug_poker_table.png ({frame.shape[1]}x{frame.shape[0]})")
                        saved_debug_img = True

                    # 1. Нарезка базовых слотов
                    cropped_data = self.cropper.crop_frame(frame)

                    # 2. Распознавание карт борда и руки
                    board_cards = (
                        self.card_detector.detect_board_cards(cropped_data.board_crops[0])
                        if cropped_data.board_crops else []
                    )

                    hero_card_l, hero_card_r, _, _, _, _ = self.card_detector.detect_hero_cards(
                        cropped_data.hero_card_crops
                    )
                    hero_cards = [c for c in (hero_card_l, hero_card_r) if c is not None]

                    # 3. Распознавание банка, стеков и ставок
                    stack_pot_res = self.stack_pot_detector.detect(frame, self.profile_dict)
                    bets = self.bet_detector.detect(frame, self.profile_dict)

                    players = []
                    seats_config = self.profile_dict.get("seats", [])

                    for seat in seats_config:
                        seat_id = seat.get("seat_id")
                        if seat_id is None:
                            continue

                        stack_val = stack_pot_res.stacks.get(seat_id, 0.0)
                        bet_val = bets.get(seat_id, 0.0)

                        status_crop = None
                        if hasattr(cropped_data, "seat_crops") and seat_id < len(cropped_data.seat_crops):
                            seat_info = cropped_data.seat_crops[seat_id]
                            if isinstance(seat_info, dict):
                                status_crop = seat_info.get("status")

                        player_status = self.status_detector.detect(
                            status_crop) if status_crop is not None else PlayerStatus.ACTIVE

                        if stack_val <= 0.0 and bet_val <= 0.0 and player_status != PlayerStatus.ALL_IN:
                            player_status = PlayerStatus.EMPTY

                        if player_status in (PlayerStatus.FOLDED, PlayerStatus.EMPTY):
                            continue

                        players.append(PlayerState(
                            seat_id=seat_id,
                            stack_bb=stack_val,
                            current_bet_bb=bet_val,
                            is_hero=(seat_id == self.hero_seat_id),
                            status=player_status
                        ))

                    # 4. Определение активного хода через TurnDetector
                    active_seat_id = self.turn_detector.detect_active_turn(cropped_data)
                    is_hero_turn = (active_seat_id == self.hero_seat_id) and (len(hero_cards) == 2)

                    # 5. Определение позиции дилера
                    detected_dealer = self.dealer_detector.detect_dealer_seat(cropped_data.seat_crops)
                    if detected_dealer is not None:
                        self._last_known_dealer_seat_id = detected_dealer

                    dealer_seat_id = getattr(self, "_last_known_dealer_seat_id", None)

                    # 6. Получение GameType
                    try:
                        game_type_enum = GameType[self.config.game_type.upper()]
                    except (KeyError, AttributeError):
                        game_type_enum = GameType.CASH_6MAX

                    # 7. Детекция турнирных данных для MTT_7MAX
                    tournament_state = None
                    if game_type_enum == GameType.MTT_7MAX:
                        tournament_crops = {}

                        # 1. Попытка вырезать турнирную зону через cropper.global_rois
                        for roi_key in ("tournament_info", "tournament_info_zone", "tournament_zone"):
                            if hasattr(self.cropper, "global_rois") and roi_key in self.cropper.global_rois:
                                roi = self.cropper.global_rois[roi_key]
                                tournament_crops['tournament_info_zone'] = self.cropper._crop(frame, roi)
                                break

                        # 2. Фоллбэк через profile_dict['global_rois']
                        if 'tournament_info_zone' not in tournament_crops and isinstance(self.profile_dict, dict):
                            global_rois = self.profile_dict.get("global_rois", {})
                            tz_bbox = (
                                global_rois.get("tournament_info")
                                or global_rois.get("tournament_info_zone")
                                or global_rois.get("tournament_zone")
                                or self.profile_dict.get("tournament_info")
                            )
                            if tz_bbox and frame.size > 0:
                                if isinstance(tz_bbox, (list, tuple)) and len(tz_bbox) == 4:
                                    x, y, w, h = tz_bbox
                                    tournament_crops['tournament_info_zone'] = frame[y:y + h, x:x + w]

                        # Передаем вырезанный кроп в детектор
                        tournament_state = self.tournament_detector.process_frame(tournament_crops)

                    # 8. Формирование полного стейта стола
                    current_stakes = getattr(self.config, "cash_stakes", None) if game_type_enum == GameType.CASH_6MAX else None

                    state = TableState(
                        game_type=game_type_enum,
                        cash_stakes=current_stakes,
                        pot_bb=stack_pot_res.pot,
                        board_cards=board_cards,
                        hero_cards=hero_cards,
                        players=players,
                        is_hero_turn=is_hero_turn,
                        dealer_seat_id=dealer_seat_id,
                        tournament_state=tournament_state
                    )

                    # 9. Расчет позиций для всех игроков
                    state.calculate_positions()

                    # 10. Запись экшена и формирование постфлоп истории
                    self.action_tracker.update(state, active_seat_id)
                    state.postflop_history = self.action_tracker.get_formatted_history()

                    # 11. Отправка стейта в UI и Solver
                    self.ui_queue.put(("STATE_UPDATE", state))

                    if state.is_hero_turn:
                        if self.state_queue.full():
                            try:
                                self.state_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.state_queue.put(state)

                except Exception as e:
                    print(f"⚠️ Ошибка в VisionThread: {e}")
                    traceback.print_exc()

                time.sleep(0.15)

    # =========================================================================
    # ПОТОК 2: SOLVER
    # =========================================================================
    def _solver_loop(self) -> None:
        while self._is_running:
            try:
                state: TableState = self.state_queue.get(timeout=0.5)

                hero_cards_str = "".join([c.to_pokerbench() for c in state.hero_cards])
                board_str = "".join([c.to_pokerbench() for c in state.board_cards])
                hand_signature = f"{hero_cards_str}_{board_str}_{state.pot_bb}"

                if self._last_processed_hand_id == hand_signature:
                    continue

                self.ui_queue.put(("SOLVER_BUSY", True))

                prompt = PokerBenchContextBuilder.build_prompt(state)
                math_info = PokerBenchContextBuilder.calculate_math(state)

                to_call = math_info.get("to_call_bb", 0.0)
                pot_odds = math_info.get("pot_odds_percent")
                pot_odds_str = f"{pot_odds:.1f}%" if pot_odds is not None else "N/A"

                print("\n" + "=" * 50)
                print("📩 ОТПРАВЛЯЕМ ПРОМПТ В OLLAMA (PokerBench):")
                print("=" * 50)
                print(prompt)
                print("-" * 50)
                print(f"📊 Доп. математика: To Call = {to_call} BB, Pot Odds = {pot_odds_str}")
                print("=" * 50 + "\n")

                decision: EngineDecision = self.solver_worker.query(
                    prompt=prompt,
                    mode=self._mode,
                    to_call_bb=math_info["to_call_bb"]
                )

                self._last_processed_hand_id = hand_signature

                self.ui_queue.put(("DECISION", decision))
                self.ui_queue.put(("SOLVER_BUSY", False))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в SolverThread: {e}")
                self.ui_queue.put(("SOLVER_BUSY", False))