import cv2
import numpy as np
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

from core.models import Card, Rank, Suit
from core.config import AppConfig


class CardDetector:
    # Маппинги рангов и мастей для парсинга шаблонов Hero карт
    RANK_MAP = {r.value.upper(): r for r in Rank}
    RANK_MAP["10"] = Rank.TEN  # Поддержка формата '10' вместо 'T'
    SUIT_MAP = {s.value.lower(): s for s in Suit}

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

        # === ИНИЦИАЛИЗАЦИЯ ДЕТЕКЦИИ HERO CARDS ===
        self.hero_left_angle = float(getattr(self.config, "hero_left_angle", -4.75))
        self.hero_right_angle = float(getattr(self.config, "hero_right_angle", 4.75))
        self.use_color = getattr(self.config, "use_color", True)
        self.active_theme = getattr(self.config, "active_theme", self.deck_name)
        self.dump_threshold = getattr(self.config, "dump_threshold", 0.70)

        base_dir = Path(__file__).resolve().parent.parent
        self.templates_dir = base_dir / "assets" / "hero_cards" / "templates" / self.active_theme
        self.unlabeled_dir = base_dir / "assets" / "unlabeled"
        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)

        self.templates: Dict[Tuple[Rank, Suit, str], np.ndarray] = {}
        self._dumped_hashes = set()

        self._load_templates()

    # =========================================================================
    # ЛОГИКА ДЕТЕКЦИИ БОРДА
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

    def _load_deck(self) -> None:
        deck_dir = Path(__file__).resolve().parent.parent / "assets" / "decks" / self.deck_name
        if not deck_dir.exists():
            print(f"⚠️ Папка с колодой не найдена: {deck_dir}")
            return

        count = 0
        for file_path in deck_dir.glob("*.png"):
            card_key = file_path.stem.lower()
            card_obj = self._parse_card_string(file_path.stem)
            if card_obj is None:
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            h, w = img.shape[:2]
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

        if isinstance(board_crop, (list, tuple)):
            slots = list(board_crop)
        else:
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

    # =========================================================================
    # ЛОГИКА ДЕТЕКЦИИ HERO CARDS
    # =========================================================================

    def _parse_filename(self, stem: str) -> Optional[Tuple[Rank, Suit, str]]:
        parts = stem.split('_')
        card_code = parts[0]

        if len(card_code) < 2:
            return None

        suit_char = card_code[-1].lower()
        rank_str = card_code[:-1].upper()

        if rank_str not in self.RANK_MAP or suit_char not in self.SUIT_MAP:
            return None

        rank_enum = self.RANK_MAP[rank_str]
        suit_enum = self.SUIT_MAP[suit_char]

        side_tag = 'any'
        if len(parts) > 1:
            tag = parts[1].lower()
            if tag in ('l', 'left'):
                side_tag = 'left'
            elif tag in ('r', 'right'):
                side_tag = 'right'

        return rank_enum, suit_enum, side_tag

    def _load_templates(self) -> None:
        if not self.templates_dir.exists():
            print(f"⚠️ Папка с шаблонами Hero не найдена: {self.templates_dir}")
            return

        read_flag = cv2.IMREAD_COLOR if self.use_color else cv2.IMREAD_GRAYSCALE

        for file_path in self.templates_dir.glob("*.png"):
            parsed = self._parse_filename(file_path.stem)
            if parsed:
                img = cv2.imread(str(file_path), read_flag)
                if img is not None:
                    self.templates[parsed] = cv2.resize(img, (32, 32))

        mode_str = "Color (BGR)" if self.use_color else "Grayscale"
        print(f"✅ Загружено эталонов Hero ({self.active_theme} | {mode_str}): {len(self.templates)} шт.")

    def _rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        if abs(angle) < 0.1:
            return img
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(
            img,
            rot_mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    def _get_card_roi(self, crop: np.ndarray, angle: float) -> np.ndarray:
        straight = self._rotate_image(crop, -angle)

        if self.use_color:
            if len(straight.shape) == 2:
                straight = cv2.cvtColor(straight, cv2.COLOR_GRAY2BGR)
            return cv2.resize(straight, (32, 32), interpolation=cv2.INTER_AREA)
        else:
            gray = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY) if len(straight.shape) == 3 else straight
            gray_norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            return cv2.resize(gray_norm, (32, 32), interpolation=cv2.INTER_AREA)

    def _save_unlabeled_crop(self, crop_32: np.ndarray, side: str, score: float) -> None:
        img_hash = hashlib.md5(crop_32.tobytes()).hexdigest()[:8]
        if img_hash in self._dumped_hashes:
            return

        self._dumped_hashes.add(img_hash)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{timestamp}_{side}_sc{score:.2f}_{img_hash}.png"
        file_path = self.unlabeled_dir / file_name

        # cv2.imwrite(str(file_path), crop_32)
        # print(f"📥 Карта сохранена в unlabeled ({self.active_theme}): {file_name}")

    def _match_card(self, crop: np.ndarray, angle: float, side: str) -> Tuple[Optional[Card], float, str]:
        if crop is None or crop.size == 0:
            return None, 0.0, "None"

        sample_32 = self._get_card_roi(crop, angle)

        if not self.templates:
            self._save_unlabeled_crop(sample_32, side, 0.0)
            return None, 0.0, "None"

        best_card_key, max_score = None, -1.0

        for (rank_enum, suit_enum, side_tag), tmpl_32 in self.templates.items():
            if side == "left" and side_tag == "right":
                continue
            if side == "right" and side_tag == "left":
                continue

            res = cv2.matchTemplate(sample_32, tmpl_32, cv2.TM_CCOEFF_NORMED)
            score = float(res[0][0])

            if score > max_score:
                max_score = score
                best_card_key = (rank_enum, suit_enum)

        if max_score < self.dump_threshold:
            self._save_unlabeled_crop(sample_32, side, max_score)

        if best_card_key and max_score >= self.dump_threshold:
            card = Card(rank=best_card_key[0], suit=best_card_key[1])
            card_str = f"{best_card_key[0].name}_{best_card_key[1].name}"
            return card, max_score, card_str

        return None, max_score, "None"

    def detect_hero_cards(
            self,
            crop_or_left: Union[np.ndarray, List[np.ndarray], Tuple[np.ndarray, ...]],
            right_crop: Optional[np.ndarray] = None
    ) -> Tuple[Optional[Card], Optional[Card], float, float, str, str]:

        left_crop = None
        if isinstance(crop_or_left, (list, tuple)):
            if len(crop_or_left) >= 2:
                left_crop, right_crop = crop_or_left[0], crop_or_left[1]
        else:
            left_crop = crop_or_left

        if left_crop is None or right_crop is None:
            return None, None, 0.0, 0.0, "None", "None"

        card_l, score_l, name_l = self._match_card(left_crop, self.hero_left_angle, side="left")
        card_r, score_r, name_r = self._match_card(right_crop, self.hero_right_angle, side="right")

        return card_l, card_r, score_l, score_r, name_l, name_r