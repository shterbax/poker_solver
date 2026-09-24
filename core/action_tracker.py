# core/action_tracker.py

from typing import Dict, List, Optional
from core.models import TableState, PlayerStatus


class ActionTracker:
    def __init__(self):
        self.current_street: str = "PREFLOP"
        self.history: Dict[str, List[str]] = {
            "PREFLOP": [],
            "FLOP": [],
            "TURN": [],
            "RIVER": []
        }
        self.last_board_count: int = 0
        self.last_hero_cards: str = ""
        self.last_active_seat: Optional[int] = None
        self.last_player_bets: Dict[int, float] = {}
        self.street_max_bet: float = 0.0

    def update(self, state: TableState, current_active_seat: Optional[int]) -> None:
        """Обновляет состояние трекера на каждом кадре."""
        hero_cards_str = "".join([c.to_pokerbench() for c in state.hero_cards])
        board_count = len(state.board_cards)

        # 1. Детекция новой раздачи (сменились карты Hero)
        if hero_cards_str and hero_cards_str != self.last_hero_cards:
            self.reset_hand(hero_cards_str)

        # 2. Детекция перехода улицы (изменилось количество карт на борде)
        if board_count != self.last_board_count:
            if board_count == 0:
                self.current_street = "PREFLOP"
            elif board_count == 3 and self.last_board_count < 3:
                self._on_street_change("FLOP")
            elif board_count == 4 and self.last_board_count < 4:
                self._on_street_change("TURN")
            elif board_count == 5 and self.last_board_count < 5:
                self._on_street_change("RIVER")

            self.last_board_count = board_count

        # 3. Фиксация действия при смене активного игрока
        if self.last_active_seat is not None and current_active_seat != self.last_active_seat:
            # Ход перешел от last_active_seat к другому игроку -> логируем действие last_active_seat
            self._record_action(self.last_active_seat, state)

        self.last_active_seat = current_active_seat

        # Обновляем максимальную ставку и кэш ставок текущей улицы
        for p in state.players:
            self.last_player_bets[p.seat_id] = p.current_bet_bb
            if p.current_bet_bb > self.street_max_bet:
                self.street_max_bet = p.current_bet_bb

    def reset_hand(self, new_hero_cards: str) -> None:
        """Сброс истории для новой раздачи."""
        self.history = {"PREFLOP": [], "FLOP": [], "TURN": [], "RIVER": []}
        self.current_street = "PREFLOP"
        self.last_board_count = 0
        self.last_hero_cards = new_hero_cards
        self.last_active_seat = None
        self.last_player_bets = {}
        self.street_max_bet = 0.0

    def _on_street_change(self, new_street: str) -> None:
        """Сброс временных параметров при выходе новой улицы."""
        self.current_street = new_street
        self.street_max_bet = 0.0
        self.last_player_bets = {}
        self.last_active_seat = None

    def _record_action(self, seat_id: int, state: TableState) -> None:
        """Определяет выполненное действие игрока по изменению размера его ставки."""
        player = next((p for p in state.players if p.seat_id == seat_id), None)

        # Получаем понятное имя позиции (например, SB, BTN, CO)
        pos_str = player.position.value if (player and player.position) else f"Seat {seat_id}"

        prev_bet = self.last_player_bets.get(seat_id, 0.0)
        curr_bet = player.current_bet_bb if player else 0.0
        curr_status = player.status if player else PlayerStatus.FOLDED

        # Логика определения действия
        if curr_status == PlayerStatus.FOLDED:
            action_text = f"{pos_str} Folds"
        elif curr_status == PlayerStatus.ALL_IN:
            action_text = f"{pos_str} All-In ({curr_bet:.1f} BB)"
        elif curr_bet > prev_bet:
            if self.street_max_bet == 0 or prev_bet == 0:
                action_text = f"{pos_str} Bets {curr_bet:.1f} BB"
            else:
                action_text = f"{pos_str} Raises to {curr_bet:.1f} BB"
        else:  # curr_bet == prev_bet
            if self.street_max_bet == 0:
                action_text = f"{pos_str} Checks"
            else:
                action_text = f"{pos_str} Calls {curr_bet:.1f} BB"

        # Избегаем дублирования идентичных записей подряд
        street_actions = self.history[self.current_street]
        if not street_actions or street_actions[-1] != action_text:
            street_actions.append(action_text)

    def get_formatted_history(self) -> str:
        """Формирует текстовый блок истории для промпта LLM."""
        formatted_sections = []
        for street in ["FLOP", "TURN", "RIVER"]:
            actions = self.history[street]
            if actions:
                formatted_sections.append(f"{street}: " + ", ".join(actions))

        return "\n".join(formatted_sections) if formatted_sections else "No postflop actions yet."