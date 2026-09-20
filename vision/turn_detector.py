import cv2
import numpy as np
from typing import List, Dict, Optional


class TurnDetector:
    def __init__(self, min_active_ratio: float = 0.03):
        """
        :param min_active_ratio: Снижен до 3%, чтобы ловить даже почти пустой таймер.
        """
        self.min_active_ratio = min_active_ratio

        # 1. Основной голубой/бирюзовый цвет активного таймера
        self.lower_cyan = np.array([80, 100, 140])
        self.upper_cyan = np.array([105, 255, 255])

        # 2. Дополнительный оранжево-красный диапазон (если таймер горит перед фолдом)
        self.lower_red1 = np.array([0, 120, 150])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 150])
        self.upper_red2 = np.array([180, 255, 255])

    def detect_active_turn(self, seat_crops: List[Dict[str, np.ndarray]]) -> Optional[int]:
        """
        Проверяет зоны timer_bar с учетом уменьшения полосы и смены цветов.
        """
        for seat_id, crops in enumerate(seat_crops):
            timer_crop = crops.get("timer_bar")
            if timer_crop is None or timer_crop.size == 0:
                continue

            hsv = cv2.cvtColor(timer_crop, cv2.COLOR_BGR2HSV)

            # Маски для разных состояний таймера
            mask_cyan = cv2.inRange(hsv, self.lower_cyan, self.upper_cyan)
            mask_red1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask_red2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)

            # Объединяем маски (любое активное состояние таймера)
            total_mask = cv2.bitwise_or(mask_cyan, mask_red1)
            total_mask = cv2.bitwise_or(total_mask, mask_red2)

            active_pixels = cv2.countNonZero(total_mask)
            total_pixels = timer_crop.shape[0] * timer_crop.shape[1]
            ratio = active_pixels / max(total_pixels, 1)

            if ratio >= self.min_active_ratio:
                return seat_id

        return None