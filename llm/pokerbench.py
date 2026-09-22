import json
import requests
from typing import List, Optional
from pydantic import BaseModel, Field

from core.models import TableState, PlayerStatus


class EngineDecision(BaseModel):
    action: str
    amount_bb: Optional[float] = None
    reasoning: Optional[str] = None


class PokerBenchContextBuilder:
    """Формирует текстовый контекст раздачи и математику для Qwen3-PokerBench."""

    @staticmethod
    def calculate_math(state: TableState) -> dict:
        hero = state.get_hero()
        if not hero:
            return {"spr": 0.0, "pot_odds_percent": 0.0, "to_call_bb": 0.0, "pot_bb": state.pot_bb}

        max_bet = max((p.current_bet_bb for p in state.players), default=0.0)
        to_call = max(0.0, max_bet - hero.current_bet_bb)

        # Расчет шансов банка с учетом суммы колла
        total_pot_for_odds = state.pot_bb + to_call
        pot_odds = (to_call / total_pot_for_odds * 100) if total_pot_for_odds > 0 else 0.0
        spr = (hero.stack_bb / state.pot_bb) if (hero.stack_bb and state.pot_bb > 0) else 0.0

        return {
            "spr": round(spr, 2),
            "pot_odds_percent": round(pot_odds, 1),
            "to_call_bb": round(to_call, 2),
            "pot_bb": round(state.pot_bb, 2)
        }

    @classmethod
    def build_prompt(cls, state: TableState, action_history_log: Optional[List[str]] = None) -> str:
        hero = state.get_hero()
        hero_pos = hero.position_label if (hero and hero.position_label) else "UNKNOWN"
        hero_stack = f"{hero.stack_bb:.1f}" if (hero and hero.stack_bb is not None) else "0.0"

        hero_hand = "".join([c.to_pokerbench() for c in state.hero_cards]) if state.hero_cards else "Unknown"
        board = "".join([c.to_pokerbench() for c in state.board_cards]) if state.board_cards else "None"

        math_info = cls.calculate_math(state)

        active_opponents = []
        for p in state.players:
            if not p.is_hero and p.status not in (PlayerStatus.FOLDED, PlayerStatus.EMPTY):
                pos = p.position_label or f"Seat_{p.seat_id}"
                stack_str = f"{p.stack_bb:.1f}" if p.stack_bb is not None else "?"
                active_opponents.append(f"{pos} ({stack_str} BB)")

        prompt_lines = [
            f"Game Type: {state.game_type.value.upper()}",
            f"Tournament: Players Left: {state.players_remaining or 'N/A'}, Avg Stack: {state.tournament_avg_stack_bb or 'N/A'} BB",
            f"Hero Position: {hero_pos} | Stack: {hero_stack} BB",
            f"Hero Hand: [{hero_hand}]",
            f"Board: [{board}]",
            f"Active Opponents: {', '.join(active_opponents) if active_opponents else 'None'}",
            "",
            "--- GAME MATH ---",
            f"Current Pot: {math_info['pot_bb']} BB",
            f"Amount to Call: {math_info['to_call_bb']} BB",
            f"Pot Odds: {math_info['pot_odds_percent']}%",
            f"Effective SPR: {math_info['spr']}",
            "",
            "--- ACTION HISTORY ---",
            "\n".join(action_history_log) if action_history_log else "Preflop action in progress...",
            "",
            "Determine the optimal GTO action for Hero."
        ]

        return "\n".join(prompt_lines)


class LLMSolverWorker:
    """Взаимодействует с локальным сервером Ollama / llama.cpp."""

    FAST_SYSTEM_PROMPT = (
        "You are an elite GTO Poker Solver Engine (Qwen3-PokerBench).\n"
        "Analyze the provided poker hand state and math carefully.\n"
        "Output ONLY a JSON object. Do not include any explanations, markdown codeblocks, or intro text.\n\n"
        "JSON Schema:\n"
        "{\n"
        '  "action": "FOLD" | "CHECK" | "CALL" | "BET" | "RAISE" | "ALL_IN",\n'
        '  "amount_bb": number or null\n'
        "}"
    )

    SMART_SYSTEM_PROMPT = (
        "You are an elite GTO Poker Coach and Analyst.\n"
        "Analyze the poker hand, range advantages, board texture, and math (Pot Odds / SPR).\n"
        "Output ONLY a JSON object containing your recommended decision and a brief reasoning (1-2 clear sentences max).\n\n"
        "JSON Schema:\n"
        "{\n"
        '  "action": "FOLD" | "CHECK" | "CALL" | "BET" | "RAISE" | "ALL_IN",\n'
        '  "amount_bb": number or null,\n'
        '  "reasoning": "string with brief GTO justification based on board texture, ranges, or equity"\n'
        "}"
    )

    def __init__(self, api_url: str = "http://localhost:11434/api/generate", model_name: str = "qwen3-4b-pokerbench"):
        self.api_url = api_url
        self.model_name = model_name

    def parse_response(self, raw_text: str, to_call_bb: float = 0.0) -> EngineDecision:
        try:
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                clean_json = raw_text[start_idx:end_idx]
                data = json.loads(clean_json)
                return EngineDecision(**data)
        except Exception as e:
            print(f"⚠️ [LLM Parser Error]: {e}")

        # Если распарсить не удалось: сбрасываем в FOLD, если против нас есть ставка, иначе в CHECK
        fallback_action = "FOLD" if to_call_bb > 0 else "CHECK"
        return EngineDecision(action=fallback_action, reasoning="Parsing Error - Fallback Decision")

    def query(self, prompt: str, mode: str = "SMART", to_call_bb: float = 0.0) -> EngineDecision:
        system_prompt = self.SMART_SYSTEM_PROMPT if mode == "SMART" else self.FAST_SYSTEM_PROMPT

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=5.0)
            if response.status_code == 200:
                raw_response = response.json().get("response", "")
                return self.parse_response(raw_response, to_call_bb)
        except Exception as e:
            print(f"❌ [LLM Request Failed]: {e}")

        fallback_action = "FOLD" if to_call_bb > 0 else "CHECK"
        return EngineDecision(action=fallback_action, reasoning="LLM Connection Timeout/Error")