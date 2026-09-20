import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.config import AppConfig
from capture.frame_cropper import FrameCropper
from vision.card_detector import CardDetector


def debug_contours_pipeline(image_name: str) -> None:
    config = AppConfig.load()

    # Автопоиск файла: сначала в tests/fixtures/, затем в tests/
    image_path = PROJECT_ROOT / "tests" / "fixtures" / image_name
    if not image_path.exists():
        image_path = PROJECT_ROOT / "tests" / image_name

    out_dir = PROJECT_ROOT / "tests" / "debug_out" / Path(image_name).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Исходный скриншот не найден: {image_path}")
        return

    cropper = FrameCropper(str(config.profile_path))
    cropped_data = cropper.crop_frame(frame)
    detector = CardDetector(config)

    if not cropped_data.hero_card_crops:
        print(f"❌ Не удалось получить кропы карт из: {image_name}")
        return

    crops = cropped_data.hero_card_crops
    left_raw = crops[0] if isinstance(crops, (list, tuple)) else crops[:, 0:crops.shape[1] // 2]
    right_raw = crops[1] if isinstance(crops, (list, tuple)) else crops[:, crops.shape[1] // 2:]

    cards = [
        ("left", left_raw, detector.hero_left_angle),
        ("right", right_raw, detector.hero_right_angle),
    ]

    for side, raw_crop, angle in cards:
        straight = detector._rotate_image(raw_crop, -angle)

        gray = cv2.cvtColor(straight, cv2.COLOR_BGR2GRAY) if len(straight.shape) == 3 else straight
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)

        vis_img = straight.copy()
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_img, w_img = binary.shape[:2]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if 6 <= h <= h_img * 0.85 and 3 <= w <= w_img * 0.85:
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            else:
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imwrite(str(out_dir / f"01_{side}_contours_boxes.png"), vis_img)
        cv2.imwrite(str(out_dir / f"02_{side}_binary_mask.png"), binary)

        glyph_32 = detector._extract_32x32_glyph(straight)
        if glyph_32 is not None:
            cv2.imwrite(str(out_dir / f"03_{side}_glyph_32x32.png"), glyph_32)
            print(f"  [+] {side.upper()}: символ вырезан и нормализован в 32x32")
        else:
            print(f"  [-] {side.upper()}: не удалось найти рамку символа")

    print(f"✅ Кадры визуализации сохранены в: {out_dir}\n")


if __name__ == "__main__":
    for img_name in ["test_hero_table1.png", "test_hero_table2.png"]:
        print(f"🔍 Отладка контуров: {img_name}")
        debug_contours_pipeline(img_name)