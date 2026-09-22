import time
import queue
import threading
from typing import Optional, Callable
import mss  # или ваш текущий инструмент захвата экрана
from capture.frame_cropper import FrameCropper
from vision.card_detector import CardDetector

from core.config import AppConfig
from core.models import TableState
from llm.pokerbench import PokerBenchContextBuilder, LLMSolverWorker, EngineDecision


class AppCore:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()

        self.cropper = FrameCropper(str(self.config.profile_path))
        self.card_detector = CardDetector(self.config)

        # Потокобезопасные очереди
        self.state_queue = queue.Queue(maxsize=1)  # Состояния от Vision к Solver
        self.ui_queue = queue.Queue()  # Сообщения для обновления CustomTkinter UI

        self.solver_worker = LLMSolverWorker()

        # Флаги управления потоками
        self._is_running = False
        self._vision_thread: Optional[threading.Thread] = None
        self._solver_thread: Optional[threading.Thread] = None

        # Кэш состояния для исключения дублирующих запросов к LLM
        self._last_processed_hand_id: Optional[str] = None
        self._mode = "SMART"  # "FAST" или "SMART"

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def start(self) -> None:
        """Запуск фоновых потоков ядра."""
        if self._is_running:
            return

        self._is_running = True

        self._vision_thread = threading.Thread(target=self._vision_loop, daemon=True, name="VisionThread")
        self._solver_thread = threading.Thread(target=self._solver_loop, daemon=True, name="SolverThread")

        self._vision_thread.start()
        self._solver_thread.start()
        print("🚀 Ядро приложения и фоновые потоки успешно запущены.")

    def stop(self) -> None:
        """Остановка фоновых процессов."""
        self._is_running = False
        print("🛑 Запрошена остановка ядра.")

    # =========================================================================
    # ПОТОК 1: COMPUTER VISION (Захват экрана и парсинг)
    # =========================================================================
    def _vision_loop(self) -> None:
        """Цикл регулярного снятия скриншота и анализа стола."""
        while self._is_running:
            try:
                # 1. Захват кадра (настройте монитор/область под окно стола)
                monitor = self.sct.monitors[1]  # Основной монитор
                sct_img = self.sct.grab(monitor)
                frame = np.array(sct_img)[:, :, :3]  # Перевод BGRA -> BGR

                # 2. Нарезка зон с помощью FrameCropper
                cropped_data = self.cropper.crop_frame(frame)

                # 3. Распознавание карт
                board_cards = self.card_detector.detect_board_cards(cropped_data.board_crops)

                hero_card_l, hero_card_r, score_l, score_r, _, _ = self.card_detector.detect_hero_cards(
                    cropped_data.hero_card_crops
                )
                hero_cards = [c for c in (hero_card_l, hero_card_r) if c is not None]

                # 4. Сборка объекта TableState
                state = TableState(
                    game_type=GameType.MTT_7MAX,  # или брать из конфига
                    pot_bb=0.0,  # Здесь подставляется результат OCR банка
                    board_cards=board_cards,
                    hero_cards=hero_cards,
                    players=[],  # Список PlayerState
                    is_hero_turn=len(hero_cards) == 2  # Пример условия хода
                )

                # 5. Отправка стейта в очереди
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

            time.sleep(0.1)  # ~10 FPS

    # =========================================================================
    # ПОТОК 2: SOLVER (Вызов LLM)
    # =========================================================================
    def _solver_loop(self) -> None:
        """Цикл ожидания ситуаций решения и обращения к LLM."""
        while self._is_running:
            try:
                # Ждем новое состояние стола, где ход принадлежит Hero
                state: TableState = self.state_queue.get(timeout=0.5)

                # Генерация хэша/идентификатора состояния раздачи, чтобы не опрашивать LLM повторно
                hero_cards_str = "".join([c.to_pokerbench() for c in state.hero_cards])
                board_str = "".join([c.to_pokerbench() for c in state.board_cards])
                hand_signature = f"{hero_cards_str}_{board_str}_{state.pot_bb}"

                if self._last_processed_hand_id == hand_signature:
                    continue  # Ход не изменился, пропуск

                # Уведомляем UI, что solver начал рассуждать
                self.ui_queue.put(("SOLVER_BUSY", True))

                # Строим промпт и считаем математику
                prompt = PokerBenchContextBuilder.build_prompt(state)
                math_info = PokerBenchContextBuilder.calculate_math(state)

                # Делаем запрос к LLM
                decision: EngineDecision = self.solver_worker.query(
                    prompt=prompt,
                    mode=self._mode,
                    to_call_bb=math_info["to_call_bb"]
                )

                self._last_processed_hand_id = hand_signature

                # Передаем готовое решение в UI
                self.ui_queue.put(("DECISION", decision))
                self.ui_queue.put(("SOLVER_BUSY", False))

            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Ошибка в SolverThread: {e}")
                self.ui_queue.put(("SOLVER_BUSY", False))