import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from core.models import PlayerStatus
from core.config import AppConfig


class StatusDetector:
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig.load()

        # Берем настройки исключительно из единого конфигуратора
        self.threshold = self.config.status_detector_threshold

        # Динамический путь на основе языка из конфига
        base_dir = Path(__file__).resolve().parent.parent
        self.templates_dir = base_dir / "assets" / "status" / self.config.language

        self.templates: Dict[PlayerStatus, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Загружает шаблоны и строго привязывает их к значениям PlayerStatus."""
        if not self.templates_dir.exists():
            print(f"⚠️ Папка с шаблонами статусов не найдена: {self.templates_dir}")
            return

        # 1. Динамически собираем все допустимые значения прямо из Enum
        status_map = {item.value: item for item in PlayerStatus}

        # 2. Алиасы на случай старых/альтернативных названий файлов (чтобы не переименовывать их на диске)
        aliases = {
            "fold": PlayerStatus.FOLDED,
            "all-in": PlayerStatus.ALL_IN
        }

        count = 0
        for file_path in self.templates_dir.glob("*.png"):
            template = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if template is None:
                continue

            name = file_path.stem.lower()

            # Ищем статус в основном словаре Enum, если нет — проверяем алиасы
            status_enum = status_map.get(name) or aliases.get(name)

            if status_enum:
                self.templates[status_enum] = template
                count += 1
            else:
                # Предупреждаем о файлах-сиротах (помогает отлавливать опечатки в названиях файлов)
                print(f"⚠️ Файл '{file_path.name}' не соответствует ни одному статусу из PlayerStatus. Пропускаем.")

        # print(f"✅ Загружено шаблонов статусов ({self.config.language}): {count}")

    def detect(self, crop: np.ndarray) -> PlayerStatus:
        """
        Сравнивает кроп статуса с эталонами. Возвращает PlayerStatus Enum.
        Если совпадений нет (нет бейджа) — считается, что игрок ACTIVE.
        """
        if crop is None or crop.size == 0:
            return PlayerStatus.EMPTY

        best_match_status = PlayerStatus.ACTIVE
        best_match_val = -1.0

        for status_enum, template in self.templates.items():
            th, tw = template.shape[:2]
            ih, iw = crop.shape[:2]

            if ih < th or iw < tw:
                continue

            result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_match_val:
                best_match_val = max_val
                best_match_status = status_enum

        if best_match_val >= self.threshold:
            return best_match_status

        # Если уверенность ниже порога (плашки Check/Call/Fold нет на экране),
        # значит игрок активно сидит в раздаче и думает.
        return PlayerStatus.ACTIVE