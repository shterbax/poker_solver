import sys
from pathlib import Path
import time
import cv2
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from capture.frame_cropper import FrameCropper
from vision.dealer_detector import DealerDetector
from core.config import AppConfig


def ensure_test_image(target_name: str = "test_table.png") -> Path:
    """Гарантирует наличие актуального изображения стола в папке tests/."""
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


def run_dealer_test(image_name: str = "test_table.png"):
    config = AppConfig.load()
    image_path = ensure_test_image(image_name)

    if not image_path.exists():
        print(f"❌ Файл {image_path} не найден! Положите скриншот стола в папку tests/")
        return

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Не удалось прочитать изображение: {image_path}")
        return

    # Загружаем кроппер и детектор с безопасными абсолютными путями
    cropper = FrameCropper(str(config.profile_path))
    template_path = PROJECT_ROOT / "assets" / "dealer_template.png"
    detector = DealerDetector(template_path=str(template_path), threshold=0.75)

    # 1. Нарезаем кадр
    cropped_data = cropper.crop_frame(frame)
    if not cropped_data.seat_crops:
        print("❌ Зоны посадочных мест (seats) не найдены в профиле!")
        return

    # 2. Измеряем скорость и точность детекции
    start_time = time.perf_counter()
    dealer_seat = detector.detect_dealer_seat(cropped_data.seat_crops)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"\n--- Тест дилера: {image_path.name} ---")
    print(f"⏱ Время детекции: {elapsed_ms:.3f} ms")
    if dealer_seat is not None:
        print(f"🎯 Фишка дилера находится у игрока S{dealer_seat}")
    else:
        print(f"❌ Дилер не найден! Проверьте {template_path.name} или подберите threshold.")


if __name__ == "__main__":
    run_dealer_test()