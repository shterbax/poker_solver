import customtkinter as ctk
from app_core import AppCore


class PokerSolverUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PokerBench GTO Solver")
        self.geometry("400x500")

        # Инициализируем и запускаем ядро
        self.core = AppCore()
        self.core.start()

        # Запускаем опрос очереди событий ядра каждые 50 мс
        self.after(50, self._process_ui_queue)

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _process_ui_queue(self):
        """Регулярно забирает обновленные данные из ядра и обновляет элементы UI."""
        while not self.core.ui_queue.empty():
            try:
                msg_type, payload = self.core.ui_queue.get_nowait()

                if msg_type == "STATE_UPDATE":
                    # Обновить отображение карт, банка и стеков
                    pass
                elif msg_type == "SOLVER_BUSY":
                    # Включить/выключить индикатор загрузки
                    pass
                elif msg_type == "DECISION":
                    # Отобразить решение (action, amount_bb, reasoning)
                    print(f"Решение: {payload.action} {payload.amount_bb} BB | {payload.reasoning}")

            except Exception:
                break

        # Планируем следующий опрос
        self.after(50, self._process_ui_queue)

    def _on_closing(self):
        self.core.stop()
        self.destroy()


if __name__ == "__main__":
    app = PokerSolverUI()
    app.mainloop()