import cv2
import numpy as np
import re
from typing import Dict

from core.ocr_engine import OcrEngine
from core.models import NormalizedROI  # Импортируем твою модель


class BetDetector:
    def __init__(self, ocr_engine: OcrEngine = None):
        self.ocr = ocr_engine or OcrEngine()
        self.cache = {}

    def _parse_bet_amount(self, raw_text: str) -> float:
        if not raw_text:
            return 0.0

        text = raw_text.lower()
        text = re.sub(r'bb|бб|\$|€|₽', '', text)

        # Нормализация частых ошибок OCR
        char_map = {
            'z': '2', 'з': '2', 'э': '2',
            's': '5', 'o': '0', 'q': '0',
            'i': '1', 'l': '1', '|': '1', ',': '.'
        }
        for char, digit in char_map.items():
            text = text.replace(char, digit)

        match = re.search(r'\d+(?:\.\d+)?', text)
        return float(match.group(0)) if match else 0.0

    def _recognize_bet(self, crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0

        # Препроцессинг для улучшения OCR
        padded = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        resized = cv2.resize(padded, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        results, _ = self.ocr.engine(resized)

        # Фолбэк: если стандартный скан не сработал, пробуем через ЧБ контраст
        if not results:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            results, _ = self.ocr.engine(bgr_gray)

        if not results:
            return 0.0

        raw_text = "".join([res[1] for res in results])
        return self._parse_bet_amount(raw_text)

    def _process_crop_with_cache(self, roi_key: str, crop: np.ndarray) -> float:
        """Сравнивает текущий кроп с кэшем. Если пиксели почти не изменились — отдаем старое значение."""
        if crop is None or crop.size == 0:
            return 0.0

        if roi_key in self.cache:
            cached_crop = self.cache[roi_key]["crop"]
            # Быстрая проверка на идентичность кадров
            if cached_crop.shape == crop.shape and cv2.absdiff(crop, cached_crop).mean() < 1.0:
                return self.cache[roi_key]["value"]

        # Если кадр обновился (фишки/цифры изменились), прогоняем OCR
        val = self._recognize_bet(crop)
        self.cache[roi_key] = {"crop": crop.copy(), "value": val}
        return val

    def detect(self, frame: np.ndarray, profile_config: dict) -> Dict[int, float]:
        """
        Возвращает плоский словарь {seat_id: bet_amount}.
        Идеально маппится на PlayerState.current_bet_bb из models.py.
        """
        results = {}
        if frame is None or frame.size == 0:
            return results

        img_h, img_w = frame.shape[:2]

        for seat in profile_config.get("seats", []):
            seat_id = seat.get("seat_id", seat.get("seat", seat.get("id")))
            if seat_id is None:
                continue

            bet_roi_raw = seat.get("bet")
            if not bet_roi_raw:
                results[seat_id] = 0.0
                continue

            # Используем NormalizedROI из core/models.py для правильной типизации и перевода
            roi = NormalizedROI(
                x=bet_roi_raw.get("x", 0.0),
                y=bet_roi_raw.get("y", 0.0),
                w=bet_roi_raw.get("w", bet_roi_raw.get("width", 0.0)),
                h=bet_roi_raw.get("h", bet_roi_raw.get("height", 0.0))
            )

            # Получаем абсолютные координаты через метод модели
            x, y, w, h = roi.to_abs(img_w, img_h)

            # Защита от выхода за границы кадра
            bet_crop = frame[max(0, y):y + h, max(0, x):x + w]

            # Сохраняем только итоговое float значение, без лишней вложенности
            results[seat_id] = self._process_crop_with_cache(f"bet_{seat_id}", bet_crop)

        return results