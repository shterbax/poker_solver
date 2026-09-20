import sys
import json
from pathlib import Path
import cv2
import os
import numpy as np

# 1. Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from capture.frame_cropper import FrameCropper
from core.ocr_engine import OcrEngine
from vision.tournament_detector import TournamentInfoCollector


def load_active_cropper() -> FrameCropper:
    """Загружает FrameCropper с профилем, указанным в config/settings.json."""
    settings_path = PROJECT_ROOT / "config" / "settings.json"
    if not settings_path.exists():
        raise FileNotFoundError(f"Файл настроек не найден: {settings_path}")

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    profile_name = settings.get("active_profile", "coinpoker_7max_mtt.json")
    profile_path = PROJECT_ROOT / "config" / "profiles" / profile_name

    if not profile_path.exists():
        raise FileNotFoundError(f"Файл профиля не найден: {profile_path}")

    print(f"Загружен активный профиль: {profile_name}")
    return FrameCropper(str(profile_path))


def crop_tournament_zone_via_cropper(cropper: FrameCropper, full_image: np.ndarray) -> np.ndarray:
    """Вырезает турнирную зону с помощью загруженного FrameCropper."""
    # 1. Проверяем наличие ключа 'tournament_info' или 'tournament_info_zone' в профиле
    if "tournament_info" in cropper.global_rois:
        roi = cropper.global_rois["tournament_info"]
        return cropper._crop(full_image, roi)
    elif "tournament_info_zone" in cropper.global_rois:
        roi = cropper.global_rois["tournament_info_zone"]
        return cropper._crop(full_image, roi)

    # 2. Безопасный фоллбэк через прямой срез массива, если ROI отсутствует в JSON
    h, w = full_image.shape[:2]
    return full_image[int(h * 0.045):int(h * 0.11), 0:int(w * 0.44)]


def test_real_ocr_on_image(image_path: str = "tests/test_table.png"):
    path = PROJECT_ROOT / image_path
    print(f"=== ЗАПУСК РЕАЛЬНОГО OCR НА ФАЙЛЕ: {path.name} ===")

    if not path.exists():
        print(f"❌ Файл не найден: {path.resolve()}")
        return

    full_image = cv2.imread(str(path))
    if full_image is None:
        print(f"❌ Не удалось прочитать изображение: {path.resolve()}")
        return

    # 1. Инициализируем кроппер на основе settings.json
    cropper = load_active_cropper()

    # 2. Вырезаем зону через cropper по координатам из профиля
    tournament_crop = crop_tournament_zone_via_cropper(cropper, full_image)

    # 3. Инициализируем OCR и коллектор
    ocr = OcrEngine()
    collector = TournamentInfoCollector(ocr)

    crops = {'tournament_info_zone': tournament_crop}

    # --- Универсальный дамп crops ---
    os.makedirs("debug_crops", exist_ok=True)

    print("\n=== ДИАГНОСТИКА CROPS ===")
    print(f"Тип переменной crops: {type(crops)}")

    if isinstance(crops, dict):
        print(f"crops — словарь с ключами: {list(crops.keys())}")
        for name, img in crops.items():
            if isinstance(img, np.ndarray) and img.size > 0:
                cv2.imwrite(f"debug_crops/{name}.png", img)
    print("=========================\n")

    state = collector.process_frame(crops)

    # 4. Выводим результаты
    print("----------------------------------------")
    print(f" Валидность (is_valid):   {state.is_valid}")
    print(f" Уровень (Level):         {state.level}")
    print(f" Блайнды (SB / BB):       {state.blinds.small_blind} / {state.blinds.big_blind}")
    print(f" Анте (Ante):             {state.blinds.ante}")
    print(f" Место в турнире:         {state.hero_rank}")
    print(f" Осталось игроков:        {state.players_remaining}")
    print(f" Средний стек (BB):       {state.avg_stack}")
    print("----------------------------------------")
    print("Результат в формате JSON:")
    print(json.dumps(state.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    test_real_ocr_on_image()