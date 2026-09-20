import sys
from pathlib import Path
import time
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from capture.frame_cropper import FrameCropper
from vision.card_detector import CardDetector
from core.config import AppConfig


def run_card_test():
    config = AppConfig.load()
    image_path = PROJECT_ROOT / "tests" / "test_table.png"

    if not config.profile_path.exists() or not image_path.exists():
        print("❌ Не найден файл профиля или тестовый скриншот!")
        return

    # Создаем папку для сохранения нарезанных слотов
    debug_dir = PROJECT_ROOT / "tests" / "test_card"
    debug_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    cropper = FrameCropper(str(config.profile_path))
    detector = CardDetector(config)

    cropped_data = cropper.crop_frame(frame)

    if cropped_data.board_crops:
        board_crop = cropped_data.board_crops[0]

        # Сохраняем исходный широкий кроп борда целиком для наглядности
        cv2.imwrite(str(debug_dir / "board_full.png"), board_crop)

        # Если борд подается единым изображением, симулируем нарезку для сохранения в цикле
        board_h, board_w = board_crop.shape[:2]
        num_slots = 5
        slot_w = board_w / num_slots

        print(f"📁 Сохранение нарезанных слотов в: {debug_dir}")
        for i in range(num_slots):
            x1 = int(i * slot_w)
            x2 = int((i + 1) * slot_w)
            slot_crop = board_crop[:, x1:x2]

            slot_path = debug_dir / f"slot_{i + 1}.png"
            cv2.imwrite(str(slot_path), slot_crop)

        start_time = time.perf_counter()
        board_cards = detector.detect_board_cards(board_crop)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"⏱ Время анализа борда: {elapsed_ms:.3f} ms")
        print(f"🎴 Распознанный борд ({len(board_cards)} карт): {[c.to_pokerbench() for c in board_cards]}")


if __name__ == "__main__":
    run_card_test()