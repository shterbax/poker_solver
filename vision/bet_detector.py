import cv2
import numpy as np
import re
import os
from core.ocr_engine import OcrEngine


class BetDetector:
    def __init__(self, ocr_engine: OcrEngine = None):
        self.ocr = ocr_engine if ocr_engine else OcrEngine()
        self.cache = {}

    def _get_pixel_roi(self, roi_dict: dict, img_w: int, img_h: int) -> tuple:
        if not roi_dict:
            return None
        x = int(roi_dict.get("x", 0) * img_w)
        y = int(roi_dict.get("y", 0) * img_h)
        w = int(roi_dict.get("w", roi_dict.get("width", 0)) * img_w)
        h = int(roi_dict.get("h", roi_dict.get("height", 0)) * img_h)
        return x, y, w, h

    def _parse_bet_amount(self, raw_text: str) -> float:
        if not raw_text:
            return 0.0

        text = raw_text.lower()
        text = re.sub(r'bb|бб|\$|€|₽', '', text)

        char_map = {
            'z': '2', 'з': '2', 'э': '2',
            's': '5', 'o': '0', 'q': '0',
            'i': '1', 'l': '1', '|': '1', ',': '.'
        }
        for char, digit in char_map.items():
            text = text.replace(char, digit)

        match = re.search(r'\d+(?:\.\d+)?', text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return 0.0

        return 0.0

    def _recognize_bet(self, crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0

        padded = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        resized = cv2.resize(padded, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        results, _ = self.ocr.engine(resized)

        if not results:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            results, _ = self.ocr.engine(bgr_gray)

        if not results:
            return 0.0

        raw_text = "".join([res[1] for res in results])
        return self._parse_bet_amount(raw_text)

    def _process_crop_with_cache(self, roi_key: str, crop: np.ndarray, dump_raw: bool) -> float:
        if crop is None or crop.size == 0:
            return 0.0

        if not dump_raw and roi_key in self.cache:
            cached_crop = self.cache[roi_key]["crop"]
            if cached_crop.shape == crop.shape:
                if cv2.absdiff(crop, cached_crop).mean() < 1.0:
                    return self.cache[roi_key]["value"]

        if dump_raw:
            os.makedirs("debug_crops", exist_ok=True)
            cv2.imwrite(f"debug_crops/{roi_key}.png", crop)

        val = self._recognize_bet(crop)

        self.cache[roi_key] = {
            "crop": crop.copy(),
            "value": val
        }

        return val

    def detect(self, frame: np.ndarray, profile_config: dict, dump_raw: bool = False) -> dict:
        results = {}
        if frame is None or frame.size == 0:
            return results

        img_h, img_w = frame.shape[:2]
        seats = profile_config.get("seats", [])

        for seat in seats:
            seat_id = seat.get("seat_id", seat.get("seat", seat.get("id")))
            if seat_id is None:
                continue

            bet_roi_dict = seat.get("bet")
            seat_result = {"bet": 0.0}

            if bet_roi_dict:
                box = self._get_pixel_roi(bet_roi_dict, img_w, img_h)
                if box:
                    x, y, w, h = box
                    bet_crop = frame[max(0, y):y + h, max(0, x):x + w]
                    if bet_crop.size > 0:
                        seat_result["bet"] = self._process_crop_with_cache(
                            f"bet_{seat_id}", bet_crop, dump_raw=dump_raw
                        )

            results[seat_id] = seat_result

        return results