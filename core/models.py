from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import numpy as np
from pydantic import BaseModel, Field


class BlindLevel(BaseModel):
    small_blind: int = 0
    big_blind: int = 0
    ante: int = 0


class TournamentState(BaseModel):
    level: Optional[int] = None
    blinds: BlindLevel = Field(default_factory=BlindLevel)
    hero_rank: Optional[int] = None          # Место в турнире
    players_remaining: Optional[int] = None  # Осталось игроков
    avg_stack: Optional[float] = None        # Средний стек (в BB)
    total_entries: Optional[int] = None
    is_valid: bool = False


class Suit(str, Enum):
    SPADES = "s"
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"


class Rank(str, Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class PlayerStatus(str, Enum):
    ACTIVE = "active"
    FOLDED = "folded"
    ALL_IN = "all_in"
    EMPTY = "empty"
    BB = "bb"
    SB = "sb"
    CALL = "call"
    BANK = "bank"
    BET = "bet"
    CHECK = "check"
    DISCONNECT = "disconnect"
    RAISE = "raise"
    STRADDLE = "straddle"


class Position(str, Enum):
    BTN = "BTN"
    SB = "SB"
    BB = "BB"
    UTG = "UTG"
    UTG1 = "UTG+1"
    MP = "MP"
    CO = "CO"


class GameType(str, Enum):
    CASH_6MAX = "6max_cash"
    MTT_7MAX = "7max_mtt"

    @property
    def max_seats(self) -> int:
        """Максимальное количество мест за столом."""
        return 6 if self == GameType.CASH_6MAX else 7

    @property
    def position_order(self) -> List[Position]:
        """
        Строгий порядок позиций по часовой стрелке, начиная с баттона (BTN = смещение 0).
        """
        if self == GameType.CASH_6MAX:
            return [
                Position.BTN,
                Position.SB,
                Position.BB,
                Position.UTG,
                Position.MP,
                Position.CO,
            ]
        elif self == GameType.MTT_7MAX:
            return [
                Position.BTN,
                Position.SB,
                Position.BB,
                Position.UTG,
                Position.UTG1,
                Position.MP,
                Position.CO,
            ]
        return []

    def get_position_for_offset(self, offset: int) -> Optional[Position]:
        """Возвращает позицию по смещению от дилера."""
        order = self.position_order
        if not order:
            return None
        return order[offset % len(order)]


@dataclass(slots=True)
class Card:
    rank: Rank
    suit: Suit

    def to_pokerbench(self) -> str:
        """Форматирование под PokerBench (например: 'Ah', 'Kd')."""
        return f"{self.rank.value}{self.suit.value}"


@dataclass(slots=True)
class NormalizedROI:
    x: float
    y: float
    w: float
    h: float

    def to_abs(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        abs_x = int(self.x * frame_w)
        abs_y = int(self.y * frame_h)
        abs_w = int(self.w * frame_w)
        abs_h = int(self.h * frame_h)
        return abs_x, abs_y, abs_w, abs_h


@dataclass
class RawCroppedFrame:
    timestamp: float
    pot_crop: Optional[np.ndarray] = None
    board_crops: List[np.ndarray] = field(default_factory=list)
    hero_card_crops: List[np.ndarray] = field(default_factory=list)
    seat_crops: List[Dict[str, np.ndarray]] = field(default_factory=list)


@dataclass
class PlayerState:
    seat_id: int
    is_hero: bool = False
    stack_bb: Optional[float] = None
    current_bet_bb: float = 0.0
    status: PlayerStatus = PlayerStatus.ACTIVE
    is_dealer: bool = False
    is_turn: bool = False
    position: Optional[Position] = None  # Использование строгого Enum вместо str


@dataclass
class TableState:
    game_type: GameType
    pot_bb: float
    board_cards: List[Card] = field(default_factory=list)
    hero_cards: List[Card] = field(default_factory=list)
    players: List[PlayerState] = field(default_factory=list)
    is_hero_turn: bool = False
    dealer_seat_id: Optional[int] = None
    # Лимит кэш-игры (опционально для MTT)
    cash_stakes: Optional[str] = None
    postflop_history: str = "No postflop actions yet."

    # Дополнительные турнирные поля (MTT)
    tournament_state: Optional[TournamentState] = None  # 👈 ДОБАВЬТЕ ЭТУ СТРОКУ
    tournament_avg_stack_bb: Optional[float] = None
    players_remaining: Optional[int] = None

    def get_hero(self) -> Optional[PlayerState]:
        for p in self.players:
            if p.is_hero:
                return p
        return None

    @property
    def hero_position(self) -> Optional[Position]:
        """Быстрый доступ к позиции Hero."""
        hero = self.get_hero()
        return hero.position if hero else None

    def calculate_positions(self) -> None:
        """
        Автоматически рассчитывает и проставляет позиции всем игрокам
        на основе dealer_seat_id и выбранного формата игры.
        """
        if self.dealer_seat_id is None:
            return

        total_seats = self.game_type.max_seats

        for player in self.players:
            # Смещение кресла по часовой стрелке относительно BTN
            offset = (player.seat_id - self.dealer_seat_id) % total_seats
            player.position = self.game_type.get_position_for_offset(offset)
            player.is_dealer = (player.seat_id == self.dealer_seat_id)