import os
import cv2
import numpy as np
from typing import List, Optional, Dict


class DealerDetector:
    def __init__(self, template_path: str = "assets/dealer_template.png", threshold: float = 0.80):
        self.threshold = threshold
        self.template, self.mask = self._load_template_with_alpha(template_path)

    def _load_template_with_alpha(self, path: str):
        if not os.path.exists(path):
            print(f"⚠️ Шаблон фишки не найден: {path}")
            return None, None

        # Читаем вместе с Alpha-каналом (UNCHANGED)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img.shape[2] == 4:
            # Разделяем BGR и Alpha-маску
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            return bgr, alpha
        return img, None

    def detect_dealer_seat(self, seat_crops: List[Dict[str, np.ndarray]]) -> Optional[int]:
        if self.template is None:
            return None

        best_seat_id = None
        max_val = -1.0

        for seat_id, crops in enumerate(seat_crops):
            dealer_crop = crops.get("dealer_btn")
            if dealer_crop is None or dealer_crop.shape[0] < self.template.shape[0] or dealer_crop.shape[1] < \
                    self.template.shape[1]:
                continue

            # Используем маску альфа-канала при наличии
            if self.mask is not None:
                res = cv2.matchTemplate(dealer_crop, self.template, cv2.TM_CCORR_NORMED, mask=self.mask)
            else:
                res = cv2.matchTemplate(dealer_crop, self.template, cv2.TM_CCOEFF_NORMED)

            _, current_max_val, _, _ = cv2.minMaxLoc(res)

            if current_max_val > max_val:
                max_val = current_max_val
                best_seat_id = seat_id

        return best_seat_id if max_val >= self.threshold else None