import re
import cv2
import numpy as np
from typing import Optional, Tuple
from core.models import TournamentState, BlindLevel
from core.blind_structure import BlindStructureManager, parse_poker_number

# Таблица замен визуально похожих латинских букв на кириллицу
HOMOGLYPH_TRANSLATION = str.maketrans({
    'a': 'а', 'A': 'а',
    'b': 'ь', 'B': 'в',
    'c': 'с', 'C': 'с',
    'e': 'е', 'E': 'е',
    'h': 'н', 'H': 'н',
    'k': 'к', 'K': 'к',
    'm': 'м', 'M': 'м',
    'o': 'о', 'O': 'о',
    'p': 'р', 'P': 'р',
    't': 'т', 'T': 'т',
    'y': 'у', 'Y': 'у',
    'x': 'х', 'X': 'х'
})


class TournamentInfoCollector:
    def __init__(self, ocr_engine):
        self.ocr = ocr_engine
        self.structure_mgr = BlindStructureManager()
        self._last_crop_hash: Optional[str] = None
        self._current_state: TournamentState = TournamentState()

        # Чистые, читаемые регулярки по нормализованному тексту
        self._level_strict_pattern = re.compile(r'(?:уровень|level)\D*?(\d{1,2})')
        self._level_fallback_pattern = re.compile(r'^\D*?(\d{1,2})\D+?\d{1,2}:\d{2}')

        self._blinds_pattern = re.compile(r'([\d\.\,]+[km]?)\s*[/|\\]\s*([\d\.\,]+[km]?)')
        self._rank_players_pattern = re.compile(r'(\d+)\s*[/|\\]\s*(\d+)')
        self._avg_stack_pattern = re.compile(r'([\d\.]+)\s*(?:bb|вв)')

    def _normalize_text(self, text: str) -> str:
        """Приводит смесь латиницы/кириллицы к единому каноническому нижнему регистру."""
        return text.translate(HOMOGLYPH_TRANSLATION).lower()

    def _compute_frame_hash(self, image: np.ndarray) -> str:
        resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        diff = gray[:, 1:] > gray[:, :-1]
        return str(hash(diff.tobytes()))

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=-20)
        scaled = cv2.resize(enhanced, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)

    def parse_level(self, text: str) -> Optional[int]:
        """Устойчивый парсинг уровня с двухэтапной защитой."""
        norm_text = self._normalize_text(text)

        # 1. Основной поиск по нормализованному слову "уровень / level"
        match = self._level_strict_pattern.search(norm_text)
        if match:
            return int(match.group(1))

        # 2. Резервный поиск: первое число перед таймером (например, "12...02:08")
        fallback_match = self._level_fallback_pattern.search(norm_text)
        if fallback_match:
            return int(fallback_match.group(1))

        return None

    def parse_blinds(self, text: str) -> Optional[Tuple[int, BlindLevel]]:
        norm_text = self._normalize_text(text)
        match = self._blinds_pattern.search(norm_text)
        if not match:
            return None

        try:
            raw_bb = match.group(2)
            parsed_bb = parse_poker_number(raw_bb)
            level_info = self.structure_mgr.match_level(parsed_bb)
            if level_info:
                blinds = BlindLevel(
                    small_blind=level_info.sb,
                    big_blind=level_info.bb,
                    ante=level_info.ante
                )
                return level_info.level, blinds
        except Exception:
            pass

        return None

    def parse_rank_and_players(self, text: str, blinds: Optional[BlindLevel] = None) -> Tuple[
        Optional[int], Optional[int]]:
        matches = self._rank_players_pattern.findall(text)
        if not matches:
            return None, None

        for r_str, p_str in matches:
            rank, players = int(r_str), int(p_str)

            if blinds and rank == blinds.small_blind and players == blinds.big_blind:
                continue

            if rank <= players:
                return rank, players

        return None, None

    def parse_avg_stack(self, text: str) -> Optional[float]:
        norm_text = self._normalize_text(text)
        match = self._avg_stack_pattern.search(norm_text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _get_level_from_manager(self, level: int):
        """Безопасное извлечение информации об уровне из BlindStructureManager."""
        mgr = self.structure_mgr

        # 1. Если есть метод get_level_info / get_by_level / get
        for method_name in ('get_level_info', 'get_by_level', 'get', 'get_level'):
            if hasattr(mgr, method_name) and callable(getattr(mgr, method_name)):
                return getattr(mgr, method_name)(level)

        # 2. Если уровни хранятся в словаре или списке (attr 'levels' или 'structure')
        for attr_name in ('levels', 'structure', 'level_map'):
            if hasattr(mgr, attr_name):
                container = getattr(mgr, attr_name)
                if isinstance(container, dict):
                    return container.get(level)
                elif isinstance(container, (list, tuple)):
                    for item in container:
                        if getattr(item, 'level', None) == level:
                            return item
        return None

    def process_frame(self, tournament_crops: dict) -> TournamentState:
        main_crop = tournament_crops.get('tournament_info_zone')
        if main_crop is None or main_crop.size == 0:
            return self._current_state

        current_hash = self._compute_frame_hash(main_crop)
        if current_hash == self._last_crop_hash:
            return self._current_state

        self._last_crop_hash = current_hash

        processed_img = self._preprocess_crop(main_crop)
        raw_text = self.ocr.image_to_string(processed_img)

        # Парсинг уровня
        level = self.parse_level(raw_text)
        blinds = None

        print(f"[DEBUG] Сырой текст OCR: {repr(raw_text)}")
        print(f"[DEBUG] Распознанный уровень: {level}")

        if level is not None:
            level_info = self._get_level_from_manager(level)
            if level_info:
                blinds = BlindLevel(
                    small_blind=level_info.sb,
                    big_blind=level_info.bb,
                    ante=level_info.ante
                )

        if level is None:
            parsed_data = self.parse_blinds(raw_text)
            if parsed_data:
                level, blinds = parsed_data

        blinds_obj = blinds or self._current_state.blinds
        hero_rank, players_rem = self.parse_rank_and_players(raw_text, blinds=blinds_obj)
        avg_stack = self.parse_avg_stack(raw_text)

        if level is not None and blinds is not None:
            current_lvl = self._current_state.level or 0
            if level >= current_lvl:
                self._current_state = TournamentState(
                    level=level,
                    blinds=blinds,
                    hero_rank=hero_rank or self._current_state.hero_rank,
                    players_remaining=players_rem or self._current_state.players_remaining,
                    avg_stack=avg_stack or self._current_state.avg_stack,
                    is_valid=True
                )

        return self._current_state