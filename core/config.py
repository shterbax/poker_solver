import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AppConfig:
    client: str = "coinpoker"
    language: str = "ru"
    active_deck: str = "full-color"
    active_profile: str = "coinpoker_6max_cash.json"
    card_detector_threshold: float = 0.75
    status_detector_threshold: float = 0.60
    dealer_detector_threshold: float = 0.80
    hero_left_angle: float = -4.75
    hero_right_angle: float = 4.75


    @property
    def profile_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "config" / "profiles" / self.active_profile

    @classmethod
    def load(cls, config_path: str = "config/settings.json") -> "AppConfig":
        """Загрузка настроек из JSON файла с фоллбеком на дефолты."""
        full_path = Path(__file__).resolve().parent.parent / config_path

        if not full_path.exists():
            print(f"⚠️ Файл настроек не найден по пути {full_path}. Создаем с дефолтными значениями.")
            config = cls()
            config.save(config_path)
            return config

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            print(f"⚠️ Ошибка чтения настроек ({e}). Используются значения по умолчанию.")
            return cls()

    def save(self, config_path: str = "config/settings.json") -> None:
        """Сохранение текущих настроек в JSON."""
        full_path = Path(__file__).resolve().parent.parent / config_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)