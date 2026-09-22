from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import numpy as np

from typing import Optional
from pydantic import BaseModel, Field

class BlindLevel(BaseModel):
    small_blind: int = 0
    big_blind: int = 0
    ante: int = 0

class TournamentState(BaseModel):
    level: Optional[int] = None
    blinds: BlindLevel = Field(default_factory=BlindLevel)
    hero_rank: Optional[int] = None          # Место в турнире (первое число дроби)
    players_remaining: Optional[int] = None  # Осталось игроков (второе число дроби)
    avg_stack: Optional[float] = None        # Средний стек (в BB)
    total_entries: Optional[int] = None
    is_valid: bool = False


class Suit(str, Enum):
    SPADES = "s"  # Пики
    HEARTS = "h"  # Черви
    DIAMONDS = "d"  # Бубны
    CLUBS = "c"  # Трефы


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
    ACTIVE = "active"  # Игрок в игре
    FOLDED = "folded"  # Сбросил карты
    ALL_IN = "all_in"  # Пошел олл-ин
    EMPTY = "empty"  # Пустое место
    BB = "bb"
    SB = "sb"
    CALL = "call"
    BANK = "bank"
    BET = "bet"
    CHECK = "check"
    DISCONNECT = "disconnect"
    RAISE = "raise"
    STRADDLE = "straddle"


class GameType(str, Enum):
    CASH_6MAX = "6max_cash"
    MTT_7MAX = "7max_mtt"


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
        """Перевод нормализованных координат (0.0-1.0) в абсолютные пиксели."""
        abs_x = int(self.x * frame_w)
        abs_y = int(self.y * frame_h)
        abs_w = int(self.w * frame_w)
        abs_h = int(self.h * frame_h)
        return abs_x, abs_y, abs_w, abs_h


@dataclass
class RawCroppedFrame:
    """Объект-контейнер для вырезанных np.ndarray кропов одного кадра."""
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
    position_label: Optional[str] = None  # BTN, SB, BB, UTG, CO...


@dataclass
class TableState:
    game_type: GameType
    pot_bb: float
    board_cards: List[Card] = field(default_factory=list)
    hero_cards: List[Card] = field(default_factory=list)
    players: List[PlayerState] = field(default_factory=list)
    is_hero_turn: bool = False
    dealer_seat_id: Optional[int] = None

    # Дополнительные турнирные поля (MTT)
    tournament_avg_stack_bb: Optional[float] = None
    players_remaining: Optional[int] = None

    def get_hero(self) -> Optional[PlayerState]:
        for p in self.players:
            if p.is_hero:
                return p
        return None