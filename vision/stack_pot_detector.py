import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from core.ocr_engine import OcrEngine
from core.models import NormalizedROI


@dataclass(slots=True)
class CacheEntry:
    """Строгий контейнер для хранения закэшированного кропа и его значения."""
    crop: np.ndarray
    value: float


@dataclass(slots=True)
class StackPotResult:
    """Строго типизированный результат распознавания."""
    pot: float = 0.0
    stacks: Dict[int, float] = field(default_factory=dict)


class StackPotDetector:
    def __init__(self, diff_threshold: float = 1.0):
        # Порог изменения пикселей для сброса кэша (1.0 защищает от микро-шума видеосжатия)
        self.diff_threshold = diff_threshold
        self.ocr = OcrEngine()
        self.cache: Dict[str, CacheEntry] = {}

    def _process_roi(
            self,
            frame: np.ndarray,
            raw_roi: Optional[Dict[str, float]],
            cache_key: str,
            dump_raw: bool
    ) -> float:
        """Универсальный метод: извлекает кроп, проверяет кэш и вызывает OCR."""
        if not raw_roi or frame is None or frame.size == 0:
            return 0.0

        img_h, img_w = frame.shape[:2]

        # Делегируем конвертацию координат датаклассу NormalizedROI
        roi = NormalizedROI(**raw_roi)
        x, y, w, h = roi.to_abs(frame_w=img_w, frame_h=img_h)
        crop = frame[y:y + h, x:x + w]

        if crop.size == 0:
            return 0.0

        # 1. Проверяем кэш на идентичность пикселей
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached.crop.shape == crop.shape:
                diff = cv2.absdiff(crop, cached.crop).mean()
                if diff < self.diff_threshold:
                    return cached.value

        # 2. Если кадр изменился, прогоняем через OCR
        if dump_raw:
            self.ocr.save_raw_crop(crop, cache_key)

        val = self.ocr.recognize(crop)

        # 3. Сохраняем в кэш (ОБЯЗАТЕЛЬНО через .copy(), чтобы отвязать память от родительского кадра)
        self.cache[cache_key] = CacheEntry(crop=crop.copy(), value=val)

        return val

    def detect(self, frame: np.ndarray, profile_config: Dict[str, Any], dump_raw: bool = False) -> StackPotResult:
        """Сканирует кадр и возвращает структурированный объект с банком и стеками."""
        result = StackPotResult()

        if frame is None or frame.size == 0:
            return result

        # 1. Извлечение банка (Pot)
        global_rois = profile_config.get("global_rois", {})
        result.pot = self._process_roi(
            frame=frame,
            raw_roi=global_rois.get("pot_amount"),
            cache_key="pot",
            dump_raw=dump_raw
        )

        # 2. Извлечение стеков игроков (Stacks)
        for seat in profile_config.get("seats", []):
            seat_id = seat.get("seat_id")
            if seat_id is not None:
                stack_val = self._process_roi(
                    frame=frame,
                    raw_roi=seat.get("stack"),
                    cache_key=f"seat_{seat_id}",
                    dump_raw=dump_raw
                )
                if stack_val > 0.0:
                    result.stacks[seat_id] = stack_val

        return result