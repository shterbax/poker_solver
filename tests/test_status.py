import sys
import cv2
import json
from pathlib import Path

# Добавляем корень проекта в пути импорта
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Модуль status_detector лежит в пакете vision
from vision.status_detector import StatusDetector


def load_table_profile():
    profile_path = BASE_DIR / "config" / "profiles" / "coinpoker_7max_mtt.json"
    if not profile_path.exists():
        profiles_dir = BASE_DIR / "config" / "profiles"
        json_files = list(profiles_dir.glob("*.json"))
        if json_files:
            profile_path = json_files[0]
        else:
            raise FileNotFoundError(f"Не найден ни один профиль стола в {profiles_dir}")

    print(f"Используем профиль стола: {profile_path.name}")
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_direct_table_detection():
    table_img_path = BASE_DIR / "tests" / "test_table.png"
    if not table_img_path.exists():
        table_img_path = BASE_DIR / "test_table.png"

    templates_dir = BASE_DIR / "assets" / "status" / "ru"

    print(f"--- Тестирование детекта по профилю для: {table_img_path.name} ---")

    table_img = cv2.imread(str(table_img_path))
    if table_img is None:
        print(f"[ERROR] Не удалось загрузить скриншот: {table_img_path}")
        return

    img_h, img_w = table_img.shape[:2]

    # Загружаем профиль
    profile = load_table_profile()
    seats = profile.get("seats", [])

    if not seats:
        print("[ERROR] В профиле стола не найдены секции 'seats'.")
        return

    # Инициализируем детектор из пакета vision
    detector = StatusDetector()

    print(f"Всего мест в профиле для проверки: {len(seats)}\n")

    for seat in seats:
        seat_id = seat.get("seat_id")
        status_roi = seat.get("status")

        if not status_roi:
            print(f"Место {seat_id}: в профиле отсутствуют координаты 'status'")
            continue

        # Переводим относительные координаты профиля (0.0 - 1.0) в пиксели изображения
        x = int(status_roi["x"] * img_w)
        y = int(status_roi["y"] * img_h)
        w = int(status_roi["w"] * img_w)
        h = int(status_roi["h"] * img_h)

        # Вырезаем область статуса из картинки стола в памяти
        status_crop = table_img[y: y + h, x: x + w]

        # Распознаем через шаблон
        detected_status = detector.detect(status_crop)

        print(f"Место {seat_id:<2} | Статус: {detected_status.upper()}")


if __name__ == "__main__":
    test_direct_table_detection()