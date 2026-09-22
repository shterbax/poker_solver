import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

from core.models import RawCroppedFrame
from core.config import AppConfig


@dataclass(frozen=True, slots=True)
class HSVRange:
    """Структура для хранения цветового диапазона в пространстве HSV."""
    lower: np.ndarray
    upper: np.ndarray


class TurnDetector:
    # Константы цветовых диапазонов таймера (Cyan/Teal + 2 оранжево-красных диапазона)
    TIMER_HSV_RANGES = (
        HSVRange(np.array([80, 100, 140]), np.array([105, 255, 255])),   # Голубой / Бирюзовый
        HSVRange(np.array([0, 120, 150]), np.array([10, 255, 255])),     # Красный / Оранжевый (1)
        HSVRange(np.array([170, 120, 150]), np.array([180, 255, 255])),   # Красный / Оранжевый (2)
    )

    def __init__(self, config: Optional[AppConfig] = None, min_active_ratio: Optional[float] = None):
        self.config = config or AppConfig.load()

        # Порог берётся из параметризации, конфигурата или задействует дефолт 3%
        self.min_active_ratio = (
            min_active_ratio
            if min_active_ratio is not None
            else getattr(self.config, "turn_detector_min_ratio", 0.03)
        )

    def _is_timer_active(self, timer_crop: np.ndarray) -> bool:
        """Вспомогательный метод: проверяет процент присутствия активного цвета таймера."""
        if timer_crop is None or timer_crop.size == 0:
            return False

        hsv = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2HSV)

        # Объединяем маски всех зарегистрированных диапазонов
        combined_mask = None
        for rng in self.TIMER_HSV_RANGES:
            mask = cv2.inRange(hsv, rng.lower, rng.upper)
            combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

        if combined_mask is None:
            return False

        active_pixels = cv2.countNonZero(combined_mask)
        total_pixels = timer_crop.shape[0] * timer_crop.shape[1]

        if total_pixels == 0:
            return False

        return (active_pixels / total_pixels) >= self.min_active_ratio

    def detect_active_turn(
        self,
        frame_data: Union[RawCroppedFrame, List[Dict[str, np.ndarray]]]
    ) -> Optional[int]:
        """
        Определяет ID игрока, чей ход сейчас активен.
        Принимает либо объект RawCroppedFrame, либо список seat_crops.
        """
        if isinstance(frame_data, RawCroppedFrame):
            seat_crops = frame_data.seat_crops
        else:
            seat_crops = frame_data

        if not seat_crops:
            return None

        for seat_id, crops in enumerate(seat_crops):
            if not isinstance(crops, dict):
                continue

            timer_crop = crops.get("timer_bar")
            if self._is_timer_active(timer_crop):
                return seat_id

        return None