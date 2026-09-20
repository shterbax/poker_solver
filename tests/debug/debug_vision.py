import sys
from pathlib import Path
import cv2
import numpy as np

# Подключаем корень проекта для доступа к модулям
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.config import AppConfig
from vision.card_detector import CardDetector
from capture.frame_cropper import FrameCropper


def debug_card_extraction(image_name: str):
    config = AppConfig.load()
    fixtures_dir = PROJECT_ROOT / "tests" / "fixtures"
    out_dir = PROJECT_ROOT / "tests" / "debug_out" / Path(image_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = fixtures_dir / image_name
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Файл не найден: {image_path}")
        return

    cropper = FrameCropper(str(config.profile_path))
    cropped = cropper.crop_frame(frame)
    detector = CardDetector(config)

    # 1. Сохраняем исходные кропы карт
    left, right = cropped.hero_card_crops[0], cropped.hero_card_crops[1]
    cv2.imwrite(str(out_dir / "01_left_raw.png"), left)
    cv2.imwrite(str(out_dir / "02_right_raw.png"), right)

    # 2. Вырезаем нормализованные символы 32x32
    rot_left = detector._rotate_image(left, -detector.hero_left_angle)
    glyph_left = detector._extract_32x32_glyph(rot_left)

    if glyph_left is not None:
        cv2.imwrite(str(out_dir / "03_left_glyph_32x32.png"), glyph_left)
        print(f"✅ [Успех] Глиф левой карты сохранен в {out_dir}")
    else:
        print(f"❌ [Ошибка] Не удалось вырезать контур символа для {image_name}")


if __name__ == "__main__":
    for test_img in ["test_hero_table1.png", "test_hero_table2.png"]:
        debug_card_extraction(test_img)