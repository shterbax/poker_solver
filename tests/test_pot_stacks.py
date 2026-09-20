import json
import os
import time
import cv2
import numpy as np
from vision.stack_pot_detector import StackPotDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_test():
    profile_path = os.path.join(
        BASE_DIR, "config", "profiles", "coinpoker_7max_mtt.json"
    )
    test_frame_path = os.path.join(BASE_DIR, "tests", "test_table.png")

    pot_dir = os.path.join(BASE_DIR, "assets", "pots")
    stack_dir = os.path.join(BASE_DIR, "assets", "stacks")

    if not os.path.exists(profile_path):
        print(f"[-] Ошибка: Конфиг {profile_path} не найден.")
        return

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    detector = StackPotDetector(pot_dir=pot_dir, stack_dir=stack_dir)

    if os.path.exists(test_frame_path):
        frame = cv2.imread(test_frame_path)
    else:
        print(
            f"[!] Тестовый кадр {test_frame_path} не найден. Создаем пустой холст 1920x1080."
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # 1. Холостой прогрев (Warmup)
    _ = detector.detect(frame, profile, dump_raw=False)

    # 2. Одиночный замер обработки
    t_start = time.perf_counter()
    results = detector.detect(frame, profile, dump_raw=True)
    single_run_ms = (time.perf_counter() - t_start) * 1000

    # 3. Серийный бенчмарк (10 итераций)
    iterations = 10
    bench_start = time.perf_counter()
    for _ in range(iterations):
        _ = detector.detect(frame, profile, dump_raw=False)
    avg_ms = ((time.perf_counter() - bench_start) * 1000) / iterations

    print("\n========================================")
    print(" РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ ")
    print("========================================")
    print(f" Total Pot : {results['pot']}")
    for seat_id, stack_val in results["stacks"].items():
        print(f" Seat {seat_id:<6} : {stack_val}")

    print("\n----------------------------------------")
    print(" МЕТРИКИ СКОРОСТИ ")
    print("----------------------------------------")
    print(f" Одиночный кадр                : {single_run_ms:.2f} ms")
    print(f" Среднее время ({iterations} итераций)  : {avg_ms:.2f} ms")
    print(f" Расчетный FPS                 : {1000 / avg_ms:.1f} FPS")
    print("========================================\n")


if __name__ == "__main__":
    run_test()