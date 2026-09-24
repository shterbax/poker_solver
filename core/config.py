import json
from dataclasses import dataclass
from pathlib import Path
from typing import List  # <--- Добавлен отсутствующий импорт

# Список всех поддерживаемых лимитов кэш-игр
CASH_STAKES_LIST: List[str] = [
    "$0.01/$0.02 (NL2)",
    "$0.02/$0.05 (NL5)",
    "$0.05/$0.10 (NL10)",
    "$0.10/$0.25 (NL25)",
    "$0.25/$0.50 (NL50)",
    "$0.50/$1.00 (NL100)",
    "$1.00/$2.00 (NL200)",
    "$2.00/$5.00 (NL500)",
    "$5.00/$10.00 (NL1000)",
    "$10.00/$20.00 (NL2000)",
    "$25.00/$50.00 (NL5000)",
]


@dataclass
class AppConfig:
    client: str = "coinpoker"
    language: str = "ru"
    active_deck: str = "full-color"
    game_type: str = "CASH_6MAX"
    active_profile: str = "coinpoker_6max_cash.json"
    cash_stakes: str = "$0.10/$0.25 (NL25)"
    card_detector_threshold: float = 0.75
    status_detector_threshold: float = 0.60
    dealer_detector_threshold: float = 0.80
    hero_left_angle: float = -4.75
    hero_right_angle: float = 4.75

    # --- НОВЫЕ ПАРАМЕТРЫ ДЛЯ OLLAMA / LLM ---
    llm_model: str = "qwen3-4b-pokerbench-grpo"
    llm_url: str = "http://localhost:11434/api/generate"

    @property
    def profile_path(self) -> Path:
        return Path(__file__).resolve().parent.parent / "config" / "profiles" / self.active_profile

    @classmethod
    def load(cls, config_path: str = "config/settings.json") -> "AppConfig":
        """Загрузка настроек с авто-объединением дефолтов из кода и JSON."""
        full_path = Path(__file__).resolve().parent.parent / config_path

        if not full_path.exists():
            config = cls()
            config.save(config_path)
            return config

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Берем свежие дефолты из класса и обновляем их значениями из JSON
            default_kwargs = cls().__dict__
            # Оставляем только те ключи из JSON, которые есть в датаклассе
            valid_user_data = {k: v for k, v in data.items() if k in default_kwargs}
            default_kwargs.update(valid_user_data)

            return cls(**default_kwargs)
        except Exception as e:
            print(f"⚠️ Ошибка чтения настроек ({e}). Используются значения по умолчанию.")
            return cls()

    def save(self, config_path: str = "config/settings.json") -> None:
        """Сохранение текущих настроек в JSON."""
        full_path = Path(__file__).resolve().parent.parent / config_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, ensure_ascii=False)