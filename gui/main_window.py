import customtkinter as ctk
from typing import Optional

from app_core import AppCore
from core.config import CASH_STAKES_LIST
from core.models import TableState
from llm.pokerbench import EngineDecision

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class PokerSolverUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PokerBench GTO Assistant")
        self.geometry("480x700")
        self.resizable(False, False)

        # Инициализация ядра
        self.core = AppCore()
        self.core.start()

        self._create_widgets()

        # Старт регулярного опроса очереди UI
        self.after(50, self._process_ui_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_widgets(self):
        # --- ВЕРХНЯЯ ПАНЕЛЬ СТАТУСА И РЕЖИМА ---
        self.header_frame = ctk.CTkFrame(self, corner_radius=10)
        self.header_frame.pack(padx=15, pady=(15, 10), fill="x")

        self.status_indicator = ctk.CTkLabel(
            self.header_frame,
            text="● Ожидание хода",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#2ECC71"
        )
        self.status_indicator.pack(side="left", padx=15, pady=10)

        self.mode_switch = ctk.CTkSegmentedButton(
            self.header_frame,
            values=["FAST", "SMART"],
            command=self._on_mode_change
        )
        self.mode_switch.set("SMART")
        self.mode_switch.pack(side="right", padx=15, pady=10)

        # --- БЛОК ТЕКУЩЕГО СОСТОЯНИЯ СТОЛА ---
        self.table_frame = ctk.CTkFrame(self, corner_radius=10)
        self.table_frame.pack(padx=15, pady=10, fill="x")

        # 1. Настройка типа стола (CASH_6MAX / MTT_7MAX)
        self.game_type_frame = ctk.CTkFrame(self.table_frame, fg_color="transparent")
        self.game_type_frame.pack(fill="x", padx=15, pady=(12, 5))

        self.game_type_label = ctk.CTkLabel(
            self.game_type_frame,
            text="Тип стола:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.game_type_label.pack(side="left")

        current_game_type = getattr(self.core.config, "game_type", "CASH_6MAX").upper()

        self.game_type_selector = ctk.CTkOptionMenu(
            self.game_type_frame,
            values=["CASH_6MAX", "MTT_7MAX"],
            command=self._on_game_type_change,
            width=140
        )
        self.game_type_selector.set(current_game_type)
        self.game_type_selector.pack(side="right")

        # 2. Настройка лимита кэш-игры (Stakes)
        self.stakes_frame = ctk.CTkFrame(self.table_frame, fg_color="transparent")
        self.stakes_frame.pack(fill="x", padx=15, pady=(5, 8))

        self.stakes_label = ctk.CTkLabel(
            self.stakes_frame,
            text="Лимит (Stakes):",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.stakes_label.pack(side="left")

        current_stake = getattr(self.core.config, "cash_stakes", "$0.10/$0.25 (NL25)")

        self.stakes_selector = ctk.CTkOptionMenu(
            self.stakes_frame,
            values=CASH_STAKES_LIST,
            command=self._on_stakes_change,
            width=200
        )
        self.stakes_selector.set(current_stake)
        self.stakes_selector.pack(side="right")

        # Если изначально выбран турнир, отключаем выбор лимита
        if current_game_type != "CASH_6MAX":
            self.stakes_selector.configure(state="disabled")

        # Карты Hero
        self.hero_cards_label = ctk.CTkLabel(
            self.table_frame,
            text="Рука: --",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.hero_cards_label.pack(anchor="w", padx=15, pady=(8, 5))

        # Карты Борда
        self.board_cards_label = ctk.CTkLabel(
            self.table_frame,
            text="Борд: --",
            font=ctk.CTkFont(size=14)
        )
        self.board_cards_label.pack(anchor="w", padx=15, pady=(0, 12))

        # --- БЛОК РЕКОМЕНДАЦИИ SOLVER ---
        self.decision_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#1E1E2E")
        self.decision_frame.pack(padx=15, pady=10, fill="both", expand=True)

        self.decision_title = ctk.CTkLabel(
            self.decision_frame,
            text="РЕКОМЕНДАЦИЯ GTO",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#89B4FA"
        )
        self.decision_title.pack(anchor="w", padx=15, pady=(15, 5))

        # Плашка с действием (FOLD / CHECK / CALL / BET / RAISE)
        self.action_badge = ctk.CTkLabel(
            self.decision_frame,
            text="WAITING...",
            font=ctk.CTkFont(size=28, weight="bold"),
            corner_radius=8,
            fg_color="#313244",
            text_color="#CDD6F4",
            height=50
        )
        self.action_badge.pack(fill="x", padx=15, pady=5)

        # Текстовое поле для логики/обоснования (Reasoning)
        self.reasoning_text = ctk.CTkTextbox(
            self.decision_frame,
            font=ctk.CTkFont(size=13),
            corner_radius=8,
            fg_color="#181825",
            text_color="#BAC2DE",
            wrap="word"
        )
        self.reasoning_text.pack(fill="both", expand=True, padx=15, pady=(10, 15))
        self.reasoning_text.insert("1.0", "Ожидание раздачи...")
        self.reasoning_text.configure(state="disabled")

    def _on_mode_change(self, selected_mode: str):
        self.core.set_mode(selected_mode)

    def _on_game_type_change(self, selected_game_type: str):
        """Вызывает смену типа стола и профиля в ядре."""
        self.core.set_game_type(selected_game_type)

        # Блокируем выбор лимитов для турниров и активируем для кэш-игр
        if selected_game_type == "CASH_6MAX":
            self.stakes_selector.configure(state="normal")
        else:
            self.stakes_selector.configure(state="disabled")

        print(f"⚙️ Тип стола изменен на {selected_game_type}")

    def _on_stakes_change(self, selected_stake: str):
        """Передает выбранный лимит в ядро."""
        self.core.set_cash_stakes(selected_stake)

    def _update_decision_badge(self, action: str, amount_bb: Optional[float]):
        action_colors = {
            "FOLD": "#E74C3C",  # Красный
            "CHECK": "#7F8C8D",  # Серый
            "CALL": "#2ECC71",  # Зеленый
            "BET": "#3498DB",  # Синий
            "RAISE": "#E67E22",  # Оранжевый
            "ALL_IN": "#9B59B6"  # Фиолетовый
        }
        color = action_colors.get(action.upper(), "#313244")

        display_text = action.upper()
        if amount_bb is not None and amount_bb > 0:
            display_text += f" {amount_bb:.1f} BB"

        self.action_badge.configure(text=display_text, fg_color=color, text_color="#FFFFFF")

    def _process_ui_queue(self):
        """Опрос сообщений из ядра."""
        while not self.core.ui_queue.empty():
            try:
                msg_type, payload = self.core.ui_queue.get_nowait()

                if msg_type == "STATE_UPDATE":
                    state: TableState = payload
                    hero_str = " ".join([c.to_pokerbench() for c in state.hero_cards]) if state.hero_cards else "--"
                    board_str = " ".join([c.to_pokerbench() for c in state.board_cards]) if state.board_cards else "--"

                    self.hero_cards_label.configure(text=f"Рука:  {hero_str}")
                    self.board_cards_label.configure(text=f"Борд:  {board_str}")

                elif msg_type == "SOLVER_BUSY":
                    is_busy: bool = payload
                    if is_busy:
                        self.status_indicator.configure(text="● Расчет GTO...", text_color="#F1C40F")
                    else:
                        self.status_indicator.configure(text="● Готово", text_color="#2ECC71")

                elif msg_type == "DECISION":
                    decision: EngineDecision = payload
                    self._update_decision_badge(decision.action, decision.amount_bb)

                    self.reasoning_text.configure(state="normal")
                    self.reasoning_text.delete("1.0", "end")
                    self.reasoning_text.insert("1.0", decision.reasoning or "Без объяснения.")
                    self.reasoning_text.configure(state="disabled")

            except Exception:
                break

        self.after(50, self._process_ui_queue)

    def _on_closing(self):
        self.core.stop()
        self.destroy()


if __name__ == "__main__":
    app = PokerSolverUI()
    app.mainloop()