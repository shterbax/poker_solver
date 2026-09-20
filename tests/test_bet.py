import json
import os
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core.ocr_engine import OcrEngine
from vision.bet_detector import BetDetector


def run_test():
    profile_path = os.path.join(BASE_DIR, "config", "profiles", "coinpoker_7max_mtt.json")
    test_frame_path = os.path.join(BASE_DIR, "tests", "test_table.png")

    if not os.path.exists(profile_path):
        print(f"[-] Ошибка: Конфиг {profile_path} не найден.")
        return

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Инициализация OCR и детектора
    ocr = OcrEngine()
    detector = BetDetector(ocr_engine=ocr)

    if os.path.exists(test_frame_path):
        frame = cv2.imread(test_frame_path)
    else:
        print(f"[!] Тестовый кадр не найден. Создаем пустой холст.")
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # 1. Холостой прогрев
    _ = detector.detect(frame, profile)

    # 2. Одиночный замер
    print("Выполняем сканирование...")
    results = detector.detect(frame, profile)

    # 3. Серийный бенчмарк кэша
    iterations = 5
    bench_start = time.perf_counter()
    for _ in range(iterations):
        _ = detector.detect(frame, profile)
    avg_ms = ((time.perf_counter() - bench_start) * 1000) / iterations

    print("\n========================================")
    # Формат вывода адаптирован под плоский словарь Dict[int, float]
    for seat_id, bet_amount in results.items():
        print(f" Seat {seat_id:<2} | Bet = {bet_amount:>8.2f}")
    print("========================================")
    print(f" Среднее время кэша ({iterations} итер.) : {avg_ms:.2f} ms")


if __name__ == "__main__":
    run_test()