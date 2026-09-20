import cv2
import numpy as np
from pathlib import Path


class StatusDetector:
    def __init__(self, templates_dir: Path = None, threshold: float = 0.60):
        """
        :param templates_dir: Путь к папке с эталонными картинками (assets/status/ru)
        :param threshold: Порог точности совпадения (0.8 = 80% сходства)
        """
        if templates_dir is None:
            # По умолчанию ищем относительно корня проекта
            BASE_DIR = Path(__file__).resolve().parent.parent
            templates_dir = BASE_DIR / "assets" / "status" / "ru"

        self.templates_dir = Path(templates_dir)
        self.threshold = threshold
        self.templates = {}

        self._load_templates()

    def _load_templates(self):
        """Загружает все шаблоны из папки assets/status/ru."""
        if not self.templates_dir.exists():
            print(f"[ERROR] Папка с шаблонами не найдена: {self.templates_dir}")
            return

        for file_path in self.templates_dir.glob("*.png"):
            template = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if template is not None:
                # Имя файла без расширения используем как название статуса (например, "fold", "check")
                status_name = file_path.stem.lower()
                self.templates[status_name] = template

    def detect(self, crop_image) -> str:
        """
        Сравнивает кроп статуса с загруженными шаблонами.
        """
        if isinstance(crop_image, (str, Path)):
            crop_image = cv2.imread(str(crop_image))

        if crop_image is None or crop_image.size == 0:
            return "empty"

        best_match_name = "unknown"
        best_match_val = 0.0

        for status_name, template in self.templates.items():
            # Шаблон должен быть меньше или равен по размеру анализируемому кропу
            th, tw = template.shape[:2]
            ih, iw = crop_image.shape[:2]
            if ih < th or iw < tw:
                continue

            # Сравнение по шаблону
            result = cv2.matchTemplate(crop_image, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_name = status_name

        # Если лучшее совпадение выше порога — возвращаем имя статуса
        if best_match_val >= self.threshold:
            return best_match_name

        return "unknown"