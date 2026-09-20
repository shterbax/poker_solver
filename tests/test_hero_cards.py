import sys
from pathlib import Path
import time
import cv2
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from capture.frame_cropper import FrameCropper
from vision.card_detector import CardDetector
from core.config import AppConfig


def ensure_test_image(target_name: str = "test_table.png") -> Path:
    """Проверяет наличие test_table.png. Если его нет, создает копию из первого доступного теста."""
    tests_dir = PROJECT_ROOT / "tests"
    target_path = tests_dir / target_name

    if not target_path.exists():
        for fallback in ["test_hero_table1.png", "test_hero_table2.png"]:
            fallback_path = tests_dir / fallback
            if fallback_path.exists():
                shutil.copy(fallback_path, target_path)
                print(f"ℹ️ Создан тестовый файл {target_name} (копия из {fallback})")
                break

    return target_path


def run_hero_test(image_name: str = "test_table.png"):
    config = AppConfig.load()
    image_path = ensure_test_image(image_name)

    if not image_path.exists():
        print(f"❌ Файл {image_path} не найден! Положите скриншот стола в папку tests/")
        return

    frame = cv2.imread(str(image_path))
    cropper = FrameCropper(str(config.profile_path))
    detector = CardDetector(config)

    cropped_data = cropper.crop_frame(frame)
    if not cropped_data.hero_card_crops:
        print("❌ Зона карт Hero не найдена в разметке профиля!")
        return

    start_time = time.perf_counter()
    card_l, card_r, score_l, score_r, name_l, name_r = detector.detect_hero_cards(cropped_data.hero_card_crops)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\n--- Тест: {image_name} ({detector.active_theme}) ---")
    print(f"⏱ Время распознавания: {elapsed_ms:.3f} ms")
    print(f"🎴 Левая карта  : {name_l} | Score: {score_l:.3f}")
    print(f"🎴 Правая карта : {name_r} | Score: {score_r:.3f}")


if __name__ == "__main__":
    run_hero_test("test_table.png")