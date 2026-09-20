import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from core.config import AppConfig
from capture.frame_cropper import FrameCropper
from vision.card_detector import CardDetector


def make_template(crop, angle, out_name):
    config = AppConfig.load()
    detector = CardDetector(config)

    # 1. Поворачиваем кроп
    straight = detector._rotate_image(crop, -angle)

    # 2. Берем фиксированный ROI ранга (верхняя левая часть)
    h, w = straight.shape[:2]
    rank_roi = straight[0:int(h * 0.45), 0:int(w * 0.50)]

    # 3. Переводим в Grayscale и приводим к 32x32
    gray = cv2.cvtColor(rank_roi, cv2.COLOR_BGR2GRAY)

    # Нормализуем контраст (Min-Max normalization)
    gray_norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    resized = cv2.resize(gray_norm, (32, 32), interpolation=cv2.INTER_AREA)

    out_dir = PROJECT_ROOT / "assets" / "decks" / "default"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / f"{out_name}.png"), resized)
    print(f" Saved template: {out_name}.png")


if __name__ == "__main__":
    config = AppConfig.load()
    cropper = FrameCropper(str(config.profile_path))

    # Дамп из table1 (Дамы)
    f1 = cv2.imread(str(PROJECT_ROOT / "tests" / "test_hero_table1.png"))
    c1 = cropper.crop_frame(f1).hero_card_crops
    make_template(c1[0], -4.75, "Q_left")

    # Дамп из table2 (10 и 7)
    f2 = cv2.imread(str(PROJECT_ROOT / "tests" / "test_hero_table2.png"))
    c2 = cropper.crop_frame(f2).hero_card_crops
    make_template(c2[0], -4.75, "10_left")
    make_template(c2[1], 4.75, "7_right")