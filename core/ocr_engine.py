import re
from typing import Optional
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OcrEngine:

    def __init__(self, templates_dir: str = None):
        self.engine = RapidOCR()

    def image_to_string(self, crop: np.ndarray) -> str:
        """Возвращает распознанный текст с изображения в виде строки."""
        if crop is None or crop.size == 0:
            return ""

        results, _ = self.engine(crop)

        if not results:
            return ""

        return "".join([res[1] for res in results])

    def recognize(self, crop: np.ndarray) -> Optional[float]:
        """Распознает сумму. Возвращает float или None."""
        if crop is None or crop.size == 0:
            return None

        results, _ = self.engine(crop)

        # Если нейросеть совсем не увидела текст (например, тусклый фон)
        if not results:
            return None

        raw_text = "".join([res[1] for res in results])

        # Исправляем частые ошибки OCR: заменяем буквы O/o на 0 до фильтрации
        raw_text = (
            raw_text.replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("l", "1")
        )

        # Очищаем от всех символов, кроме цифр, точек и запятых
        clean_text = re.sub(r"[^\d.,]", "", raw_text)

        if not clean_text:
            return None

        try:
            if "," in clean_text and "." in clean_text:
                clean_text = clean_text.replace(",", "")
            elif clean_text.count(",") == 1 and "." not in clean_text:
                clean_text = clean_text.replace(",", ".")
            else:
                clean_text = clean_text.replace(",", "")

            return float(clean_text)
        except ValueError:
            return None

    def save_raw_crop(self, crop: np.ndarray, prefix: str) -> bool:
        return True