import sys
import time
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from capture.frame_cropper import FrameCropper
from vision.turn_detector import TurnDetector
from core.config import AppConfig


def run_turn_test():
    config = AppConfig.load()
    profile_path = PROJECT_ROOT / "config" / "profiles" / config.active_profile
    image_path = PROJECT_ROOT / "tests" / "test_table.png"

    if not profile_path.exists() or not image_path.exists():
        print("❌ Файлы профиля или скриншота не найдены!")
        return

    frame = cv2.imread(str(image_path))
    cropper = FrameCropper(str(profile_path))
    turn_detector = TurnDetector(config=config)

    # Захват кадра возвращает контейнер RawCroppedFrame
    cropped_data = cropper.crop_frame(frame)

    start_time = time.perf_counter()
    # Теперь передавать можно прямо весь cropped_data
    active_seat = turn_detector.detect_active_turn(cropped_data)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"⏱ Время анализа ходов: {elapsed_ms:.4f} ms")
    if active_seat is not None:
        print(f"🟢 Сейчас ход игрока S{active_seat}")
    else:
        print("⚪ Ни один из игроков сейчас не ходит (или пауза между раздачами)")


if __name__ == "__main__":
    run_turn_test()