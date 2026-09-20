import cv2
import numpy as np
from core.ocr_engine import OcrEngine


class StackPotDetector:
    def __init__(self, pot_dir: str = "assets/pots", stack_dir: str = "assets/stacks"):
        # Единый экземпляр OCR для всех задач экономит ресурсы
        self.ocr = OcrEngine()

        # Хранилище состояний: { "roi_name": {"crop": np.ndarray, "value": float} }
        self.cache = {}

    def _get_pixel_roi(self, roi_dict: dict, img_w: int, img_h: int) -> tuple:
        """Переводит относительные координаты (0.0-1.0) в абсолютные пиксели."""
        if not roi_dict:
            return None
        x = int(roi_dict["x"] * img_w)
        y = int(roi_dict["y"] * img_h)
        w = int(roi_dict["w"] * img_w)
        h = int(roi_dict["h"] * img_h)
        return x, y, w, h

    def _process_crop_with_cache(self, roi_key: str, crop: np.ndarray, dump_raw: bool) -> float:
        """Сравнивает пиксели с прошлым кадром. Если изменений нет, отдает кэш."""
        if crop is None or crop.size == 0:
            return 0.0

        # 1. Проверяем наличие ROI в кэше
        if roi_key in self.cache:
            cached_crop = self.cache[roi_key]["crop"]

            # Проверяем совпадение размеров
            if cached_crop.shape == crop.shape:
                # Вычисляем разницу пикселей (absdiff)
                # Порог 1.0 защищает от микро-шума видеосжатия, если берется захват экрана
                diff = cv2.absdiff(crop, cached_crop).mean()
                if diff < 1.0:
                    return self.cache[roi_key]["value"]

        # 2. Если пиксели изменились — прогоняем через нейросеть
        if dump_raw:
            self.ocr.save_raw_crop(crop, roi_key)

        val = self.ocr.recognize(crop)

        # 3. Обновляем кэш. Обязательно используем .copy(),
        # чтобы отвязать срез памяти от родительского кадра.
        self.cache[roi_key] = {
            "crop": crop.copy(),
            "value": val
        }

        return val

    def detect(self, frame: np.ndarray, profile_config: dict, dump_raw: bool = False) -> dict:
        results = {
            "pot": 0.0,
            "stacks": {}
        }

        if frame is None or frame.size == 0:
            return results

        img_h, img_w = frame.shape[:2]

        # 1. Извлечение Pot
        global_rois = profile_config.get("global_rois", {})
        pot_roi_dict = global_rois.get("pot_amount")

        if pot_roi_dict:
            x, y, w, h = self._get_pixel_roi(pot_roi_dict, img_w, img_h)
            pot_crop = frame[y:y + h, x:x + w]
            if pot_crop.size > 0:
                results["pot"] = self._process_crop_with_cache("pot", pot_crop, dump_raw)

        # 2. Извлечение стеков игроков
        seats = profile_config.get("seats", [])
        for seat in seats:
            seat_id = seat.get("seat_id")
            stack_roi_dict = seat.get("stack")

            if seat_id is not None and stack_roi_dict:
                x, y, w, h = self._get_pixel_roi(stack_roi_dict, img_w, img_h)
                stack_crop = frame[y:y + h, x:x + w]
                if stack_crop.size > 0:
                    roi_key = f"seat_{seat_id}"
                    results["stacks"][seat_id] = self._process_crop_with_cache(roi_key, stack_crop, dump_raw)

        return results