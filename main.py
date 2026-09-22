import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from gui.main_window import PokerSolverUI


def main():
    print("🚀 Запуск PokerBench Solver...")
    app = PokerSolverUI()
    app.mainloop()


if __name__ == "__main__":
    main()