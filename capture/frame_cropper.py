import json
import time
from typing import Dict, Any
import numpy as np

from core.models import NormalizedROI, RawCroppedFrame


class FrameCropper:
    def __init__(self, profile_path: str):
        self.profile_data: Dict[str, Any] = {}
        self.global_rois: Dict[str, NormalizedROI] = {}
        self.seats_rois: list[Dict[str, NormalizedROI]] = []

        self._load_profile(profile_path)

    def _load_profile(self, path: str) -> None:
        """Загрузка JSON-профиля калибровки и инициализация ROI моделей."""
        with open(path, "r", encoding="utf-8") as f:
            self.profile_data = json.load(f)

        # Парсим глобальные зоны
        for name, coords in self.profile_data.get("global_rois", {}).items():
            self.global_rois[name] = NormalizedROI(**coords)

        # Парсим зоны для каждого игрока
        for seat in self.profile_data.get("seats", []):
            seat_map = {}
            for key, val in seat.items():
                if key != "seat_id" and isinstance(val, dict):
                    seat_map[key] = NormalizedROI(**val)
            self.seats_rois.append(seat_map)

    @staticmethod
    def _crop(frame: np.ndarray, roi: NormalizedROI) -> np.ndarray:
        """Быстрый клейсинг NumPy-массива по нормализованным координатам."""
        h, w = frame.shape[:2]
        abs_x, abs_y, abs_w, abs_h = roi.to_abs(w, h)

        # NumPy срезы: [y:y+h, x:x+w]
        return frame[abs_y: abs_y + abs_h, abs_x: abs_x + abs_w]

    def crop_frame(self, frame: np.ndarray) -> RawCroppedFrame:
        """
        Нарезает входной кадр стола на изолированные зоны.
        Время выполнения: < 0.2 ms
        """
        timestamp = time.time()

        # 1. Глобальные зоны
        pot_crop = None
        if "pot_amount" in self.global_rois:
            pot_crop = self._crop(frame, self.global_rois["pot_amount"])

        # Зона борда
        board_crops = []
        if "board_cards" in self.global_rois:
            board_crops.append(self._crop(frame, self.global_rois["board_cards"]))

        # Зоны карт Hero (поддержка раздельных hero_left/hero_right и единой зоны hero_cards)
        hero_card_crops = []
        if "hero_left" in self.global_rois and "hero_right" in self.global_rois:
            hero_card_crops.append(self._crop(frame, self.global_rois["hero_left"]))
            hero_card_crops.append(self._crop(frame, self.global_rois["hero_right"]))
        elif "hero_cards" in self.global_rois:
            hero_card_crops.append(self._crop(frame, self.global_rois["hero_cards"]))

        # 2. Зоны игроков
        seat_crops = []
        for seat_map in self.seats_rois:
            cropped_seat_fields = {}
            for field_name, roi in seat_map.items():
                cropped_seat_fields[field_name] = self._crop(frame, roi)
            seat_crops.append(cropped_seat_fields)

        return RawCroppedFrame(
            timestamp=timestamp,
            pot_crop=pot_crop,
            board_crops=board_crops,
            hero_card_crops=hero_card_crops,
            seat_crops=seat_crops
        )