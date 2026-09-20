import cv2
import os
from capture.frame_cropper import FrameCropper


def run_test():
    profile_path = "config/profiles/coinpoker_6max_cash.json"  # Укажите путь к вашему JSON
    image_path = "tests/test_table.png"  # Укажите путь к тестовому скриншоту

    if not os.path.exists(profile_path) or not os.path.exists(image_path):
        print("❌ Файлы для теста не найдены!")
        return

    # Загружаем кадр через OpenCV
    frame = cv2.imread(image_path)
    cropper = FrameCropper(profile_path)

    # Нарезаем
    cropped_data = cropper.crop_frame(frame)

    print("✅ Кадр успешно нарезан!")
    print(f"Вырезано мест: {len(cropped_data.seat_crops)}")

    # Сохраним кроп банка для проверки
    if cropped_data.pot_crop is not None:
        cv2.imwrite("tests/debug_pot.png", cropped_data.pot_crop)
        print("💾 Кроп банка сохранен в tests/debug_pot.png")


if __name__ == "__main__":
    run_test()