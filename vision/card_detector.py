import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

from core.models import Card, Rank, Suit
from core.config import AppConfig


class CardDetector:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()

        # === ИНИЦИАЛИЗАЦИЯ ДЕТЕКЦИИ БОРДА ===
        self.deck_name = self.config.active_deck
        self.threshold = float(self.config.card_detector_threshold)

        self.raw_corners: Dict[str, np.ndarray] = {}
        self.card_objects: Dict[str, Card] = {}
        self.cached_board_h: Optional[int] = None
        self.scaled_board_templates: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

        self._load_deck()

    # =========================================================================
    # УНИВЕРСАЛЬНЫЙ ПАРСИНГ ЧЕРЕЗ ENUM МОДЕЛЕЙ
    # =========================================================================

    def _parse_card_string(self, card_str: str) -> Optional[Card]:
        """Парсит строку (например, 'Qc', '10d') в объект Card через базовые Enums."""
        card_str = card_str.upper().replace('10', 'T')
        if len(card_str) < 2:
            return None

        rank_str = card_str[:-1]
        suit_str = card_str[-1].lower()

        try:
            return Card(rank=Rank(rank_str), suit=Suit(suit_str))
        except ValueError:
            return None

    # =========================================================================
    # ЛОГИКА ДЕТЕКЦИИ БОРДА
    # =========================================================================

    def _load_deck(self) -> None:
        deck_dir = Path(__file__).resolve().parent.parent / "assets" / "decks" / self.deck_name
        if not deck_dir.exists():
            print(f"⚠️ Папка с колодой не найдена: {deck_dir}")
            return

        count = 0
        for file_path in deck_dir.glob("*.png"):
            card_key = file_path.stem.lower()  # Приводим к нижнему регистру для ключа
            card_obj = self._parse_card_string(file_path.stem)
            if card_obj is None:
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
            # Берем левый верхний угол карты (ранг + масть)
            corner = img[0:int(h * 0.45), 0:int(w * 0.45)]

            self.raw_corners[card_key] = corner
            self.card_objects[card_key] = card_obj
            count += 1


    def _update_scaled_cache(self, target_board_h: int) -> None:
        if self.cached_board_h == target_board_h:
            return

        self.scaled_board_templates.clear()
        desired_corner_h = max(10, int(target_board_h * 0.42))

        for card_key, raw_corner in self.raw_corners.items():
            tmpl_h, tmpl_w = raw_corner.shape[:2]
            scale = desired_corner_h / tmpl_h
            target_w = max(5, int(tmpl_w * scale))

            resized = cv2.resize(raw_corner, (target_w, desired_corner_h), interpolation=cv2.INTER_AREA)

            if resized.shape[2] == 4:
                bgr = resized[:, :, :3]
                mask = resized[:, :, 3]
            else:
                bgr = resized
                mask = None

            self.scaled_board_templates[card_key] = (bgr, mask)

        self.cached_board_h = target_board_h

    @staticmethod
    def _is_slot_empty(slot_crop: np.ndarray) -> bool:
        """Быстрый фильтр: проверяет, есть ли вообще карта в слоте (с учетом темных тем)."""
        if slot_crop is None or slot_crop.size == 0:
            return True

        gray = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray)
        max_val = np.max(gray)

        if std_dev < 15.0 or max_val < 100:
            return True

        return False

    def _detect_in_slot(self, slot_crop: np.ndarray) -> Optional[Card]:
        if self._is_slot_empty(slot_crop):
            return None

        best_card_key = None
        max_val = -1.0

        for card_key, (template, mask) in self.scaled_board_templates.items():
            if slot_crop.shape[0] < template.shape[0] or slot_crop.shape[1] < template.shape[1]:
                continue

            method = cv2.TM_CCORR_NORMED if mask is not None else cv2.TM_CCOEFF_NORMED
            res = cv2.matchTemplate(slot_crop, template, method, mask=mask)
            _, current_max_val, _, _ = cv2.minMaxLoc(res)

            if current_max_val > max_val:
                max_val = current_max_val
                best_card_key = card_key

        dynamic_threshold = max(0.88, self.threshold)

        if max_val >= dynamic_threshold and best_card_key in self.card_objects:
            return self.card_objects[best_card_key]

        return None

    def detect_board_cards(
            self,
            board_crop: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]]
    ) -> List[Card]:
        if board_crop is None:
            return []

        # 1. Если передан готовый список/кортеж отдельных слотов
        if isinstance(board_crop, (list, tuple)):
            slots = list(board_crop)
        else:
            # 2. Если передан единый широкий снимок борда — нарезаем его на 5 частей
            if not isinstance(board_crop, np.ndarray) or board_crop.size == 0:
                return []

            board_h, board_w = board_crop.shape[:2]
            num_slots = 5
            slot_w = board_w / num_slots

            slots = []
            for i in range(num_slots):
                x1 = int(i * slot_w)
                x2 = int((i + 1) * slot_w)
                slots.append(board_crop[:, x1:x2])

        detected_cards = []
        for slot in slots:
            if slot is None or not isinstance(slot, np.ndarray) or slot.size == 0:
                continue
            self._update_scaled_cache(slot.shape[0])
            card = self._detect_in_slot(slot)
            if card:
                detected_cards.append(card)

        return detected_cards

    def detect_board(
            self,
            board_crop: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]]
    ) -> List[Card]:
        return self.detect_board_cards(board_crop)
