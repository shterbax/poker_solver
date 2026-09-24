import json
import requests
from typing import List, Optional
from pydantic import BaseModel, Field
import time
from core.logger import DecisionLogger

from core.models import TableState, GameType, Position, PlayerStatus


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

    @staticmethod
    def _build_preflop_history(state: TableState) -> List[str]:
        """Восстанавливает хронологию экшена на префлопе по ставкам и позициям игроков."""
        preflop_order = ["UTG", "MP", "MP1", "MP2", "HJ", "CO", "BTN", "SB", "BB"]

        # Индексируем активных игроков по строковому представлению позиции
        players_by_pos = {}
        for p in state.players:
            if p.position:
                pos_str = p.position.value if hasattr(p.position, 'value') else str(p.position)
                players_by_pos[pos_str] = p

        history = []
        raise_count = 0
        current_max_bet = 1.0  # Базовый размер ББ

        for pos in preflop_order:
            if pos not in players_by_pos:
                continue

            player = players_by_pos[pos]
            bet = player.current_bet_bb
            is_all_in = (player.status == PlayerStatus.ALL_IN)
            all_in_str = " All-in" if is_all_in else ""

            # Пропускаем стандартную проставку блайндов без хода
            if pos == "SB" and bet <= 0.5 and raise_count == 0:
                continue
            if pos == "BB" and bet <= 1.0 and raise_count == 0:
                continue

            # 1. Если ставка превышает текущий максимум -> это Рейз / 3-бет / 4-бет
            if bet > current_max_bet:
                raise_count += 1
                current_max_bet = bet

                if raise_count == 1:
                    history.append(f"{pos} raises{all_in_str} to {bet:.1f} BB")
                elif raise_count == 2:
                    history.append(f"{pos} re-raises (3-bet){all_in_str} to {bet:.1f} BB")
                elif raise_count == 3:
                    history.append(f"{pos} re-raises (4-bet){all_in_str} to {bet:.1f} BB")
                else:
                    history.append(f"{pos} re-raises ({raise_count}-bet){all_in_str} to {bet:.1f} BB")

            # 2. Если ставка равна текущей максимальной (> 1 BB) -> это Колл
            elif bet == current_max_bet and bet > 1.0:
                history.append(f"{pos} calls{all_in_str} {bet:.1f} BB")

            # 3. Если ставка равна 1 BB от свободной позиции -> это Лимп
            elif bet == 1.0 and pos not in ("SB", "BB") and raise_count == 0:
                history.append(f"{pos} limps (1.0 BB)")

        # Контекст позиции Hero
        hero_pos_str = state.hero_position.value if state.hero_position else "UNKNOWN"

        if raise_count == 0:
            history.append(f"Hero is on {hero_pos_str} facing Limps/Unopened pot")
        else:
            facing_action = "raise" if raise_count == 1 else f"{raise_count}-bet"
            history.append(f"Hero is on {hero_pos_str} facing {facing_action}")

        return history

    @staticmethod
    def build_prompt(state: TableState) -> str:
        hero = state.get_hero()
        hero_stack = hero.stack_bb if (hero and hero.stack_bb is not None) else 0.0

        # 1. Позиция Hero
        hero_pos_str = state.hero_position.value if state.hero_position else "UNKNOWN"

        hero_cards_str = " ".join([c.to_pokerbench() for c in state.hero_cards])
        board_str = " ".join([c.to_pokerbench() for c in state.board_cards]) if state.board_cards else "None"

        # --- ФОРМИРОВАНИЕ ШАПКИ С УЧЕТОМ ЛИМИТА ---
        if state.game_type == GameType.CASH_6MAX and state.cash_stakes:
            prompt_lines = [f"Game Type: {state.game_type.name} | Stakes: {state.cash_stakes}"]
        else:
            prompt_lines = [f"Game Type: {state.game_type.name}"]

        # --- ОБРАБОТКА ТУРНИРНЫХ ДАННЫХ (MTT) ---
        if state.game_type == GameType.MTT_7MAX:
            ts = state.tournament_state
            t_parts = []

            if ts:
                if ts.level is not None:
                    t_parts.append(f"Level {ts.level}")
                if ts.blinds and ts.blinds.big_blind > 0:
                    t_parts.append(f"Blinds {ts.blinds.small_blind}/{ts.blinds.big_blind} (Ante {ts.blinds.ante})")
                if ts.hero_rank and ts.players_remaining:
                    t_parts.append(f"Rank {ts.hero_rank}/{ts.players_remaining}")
                elif ts.players_remaining:
                    t_parts.append(f"Players Left: {ts.players_remaining}")
                elif state.players_remaining:
                    t_parts.append(f"Players Left: {state.players_remaining}")

                avg_val = ts.avg_stack if ts.avg_stack is not None else state.tournament_avg_stack_bb
                if avg_val:
                    t_parts.append(f"Avg Stack: {avg_val:.1f} BB")
            elif state.players_remaining is not None:
                t_parts.append(f"Players Left: {state.players_remaining}")
                if state.tournament_avg_stack_bb:
                    t_parts.append(f"Avg Stack: {state.tournament_avg_stack_bb:.1f} BB")

            if t_parts:
                prompt_lines.append(f"Tournament Context: {', '.join(t_parts)}")

        hero_bet_str = f" | Bet: {hero.current_bet_bb:.1f} BB" if (hero and hero.current_bet_bb > 0) else ""
        prompt_lines.extend([
            f"Hero Position: {hero_pos_str} | Stack: {hero_stack:.1f} BB{hero_bet_str}",
            f"Hero Hand: [{hero_cards_str}]",
            f"Board: [{board_str}]"
        ])

        # 2. Формирование списка оппонентов
        opponents_info = []
        for p in state.players:
            if p.is_hero:
                continue

            pos_name = p.position.value if p.position else f"Seat_{p.seat_id}"
            stack_val = p.stack_bb if p.stack_bb is not None else 0.0

            bet_str = f", Bet: {p.current_bet_bb:.1f} BB" if p.current_bet_bb > 0 else ""

            status_str = ""
            if p.status and p.status not in (PlayerStatus.ACTIVE, PlayerStatus.EMPTY):
                status_str = f", Status: {p.status.value}"

            opponents_info.append(f"{pos_name} ({stack_val:.1f} BB{bet_str}{status_str})")

        opponents_str = ", ".join(opponents_info) if opponents_info else "None"
        prompt_lines.append(f"Active Opponents: {opponents_str}")

        # 3. Динамическое определение текущей улицы
        board_count = len(state.board_cards) if state.board_cards else 0
        if board_count == 0:
            street_name = "Preflop"
        elif board_count == 3:
            street_name = "Flop"
        elif board_count == 4:
            street_name = "Turn"
        elif board_count == 5:
            street_name = "River"
        else:
            street_name = "Postflop"

        # Дополнительная математика
        math_data = PokerBenchContextBuilder.calculate_math(state)

        # Формирование истории действий (Префлоп + Постфлоп)
        if street_name == "Preflop":
            action_lines = PokerBenchContextBuilder._build_preflop_history(state)
            action_history_str = "\n".join(action_lines)
        else:
            preflop_lines = PokerBenchContextBuilder._build_preflop_history(state)
            preflop_summary = "Preflop Summary: " + " -> ".join(preflop_lines)
            postflop_actions = getattr(state, "postflop_history", "No postflop actions yet.")
            action_history_str = f"{preflop_summary}\n\nPostflop Action History:\n{postflop_actions}"

        prompt_lines.extend([
            "",
            "--- GAME MATH ---",
            f"Current Pot: {state.pot_bb:.2f} BB",
            f"Amount to Call: {math_data.get('to_call_bb', 0.0):.2f} BB",
            f"Pot Odds: {math_data.get('pot_odds_percent', 0.0):.1f}%",
            f"Effective SPR: {math_data.get('spr', 0.0):.2f}",
            "",
            "--- ACTION HISTORY ---",
            f"Current Street: {street_name}",
            action_history_str,
            "",
            "Determine the optimal GTO action for Hero."
        ])

        return "\n".join(prompt_lines)


class LLMSolverWorker:
    """Взаимодействует с локальным сервером Ollama / llama.cpp."""

    FAST_SYSTEM_PROMPT = """"You are a strict GTO poker bot playing 6-max NLHE Cash Games.
        You must analyze the provided game state and output your decision STRICTLY in valid JSON format. No explanations, no markdown formatting, no comments.
        
        CRITICAL RULES:
        1. ACTION MAPPING:
           - If "Amount to Call" == 0.0: You can ONLY choose "CHECK" or "RAISE". NEVER choose "FOLD" or "CALL".
           - If "Amount to Call" > 0.0: You can ONLY choose "FOLD", "CALL", or "RAISE". NEVER choose "CHECK".
        2. EXTREME AGGRESSION (>50 BB): NEVER call massive All-Ins or bets larger than 50 BB unless you hold an absolute premium hand (QQ, KK, AA, or AK). Fold all other pairs, draws, and marginal hands.
        3. RAISE SIZING GTO:
           - Preflop Open: Always 3.0 BB.
           - Preflop 3-Bet/4-Bet: Always 3x the previous bet.
           - Postflop Raise: Base it on pot size (e.g., 33%, 50%, 75% of total pot).
           - Minimum Raise Rule: If action="RAISE", amount_bb MUST be at least 2.0 BB. Never use micro-raises like 0.5 BB.
        4. DEPENDENT SIZING:
           - If action="CALL", amount_bb MUST exactly match "Amount to Call".
           - If action="FOLD" or "CHECK", amount_bb MUST be exactly 0.0.   
            
        OUTPUT TEMPLATE:
        {
          "action": "FOLD",
          "amount_bb": 0.0
        }"""

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

    def __init__(self, api_url: str = "http://localhost:11434/api/generate", model_name: str = "qwen3-4b-pokerbench-grpo"):
        self.api_url = api_url
        self.model_name = model_name
        self.logger = DecisionLogger()

    def parse_response(self, raw_text: str, to_call_bb: float = 0.0) -> EngineDecision:
        try:
            if "</think>" in raw_text:
                raw_text = raw_text.split("</think>")[-1]

            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                clean_json = raw_text[start_idx:end_idx]
                data = json.loads(clean_json)

                action_raw = data.get("action") or data.get("action_string") or data.get("action_short") or ""
                action_raw = str(action_raw).lower().strip()
                words = action_raw.replace("_", " ").replace("-", " ").split()

                final_action = "CHECK"

                # Парсинг действия
                if any(w in words for w in ["fold", "f"]):
                    final_action = "FOLD"
                    # 🚨 Защита от бесплатных фолдов
                    if to_call_bb == 0.0:
                        final_action = "CHECK"
                elif any(w in words for w in ["call", "c", "cc"]):
                    final_action = "CALL"
                elif any(w in words for w in ["raise", "bet", "cbr", "r"]):
                    final_action = "RAISE"
                elif any(w in words for w in ["check", "chk"]):
                    final_action = "CHECK"
                else:
                    final_action = "FOLD" if to_call_bb > 0 else "CHECK"

                amount = float(data.get("amount_bb") or 0.0)

                # Обнуляем сумму для FOLD и CHECK
                if final_action in ["FOLD", "CHECK"]:
                    amount = 0.0

                # 🚨 Защита от неадекватных микро-рейзов (галлюцинаций сайзинга)
                if final_action == "RAISE":
                    min_raise = max(2.0, to_call_bb + 1.0)
                    if amount < min_raise:
                        amount = float(min_raise)

                return EngineDecision(action=final_action, amount_bb=amount)

        except Exception as e:
            print(f"⚠️ [LLM Parser Error]: {e} | Raw: {raw_text}")

        # Fallback
        fallback_action = "FOLD" if to_call_bb > 0 else "CHECK"
        return EngineDecision(action=fallback_action, amount_bb=0.0, reasoning="Parsing Error - Fallback Decision")



        # Fallback при ошибке JSON
        fallback_action = "FOLD" if to_call_bb > 0 else "CHECK"
        return EngineDecision(action=fallback_action, amount_bb=0.0, reasoning="Parsing Error - Fallback Decision")

    def query(self, state=None, prompt: str = "", mode: str = "FAST", to_call_bb: float = 0.0,
              **kwargs) -> EngineDecision:
        """
        Универсальный метод query, защищенный от любых вариантов вызова из SolverThread.
        """
        # 1. Если первым позиционным аргументом случайно передали текст промпта вместо state
        if isinstance(state, str) and not prompt:
            prompt = state
            state = kwargs.get("state", None)

        # 2. Перехватываем значение prompt, если оно пришло в kwargs
        if not prompt and "prompt" in kwargs:
            prompt = kwargs["prompt"]

        # 3. Перехватываем mode и to_call_bb из kwargs
        mode = kwargs.get("mode", mode)
        to_call_bb = kwargs.get("to_call_bb", to_call_bb)

        # Вычисляем to_call_bb из state, если он не был передан явно
        if to_call_bb == 0.0 and state and hasattr(state, "get_hero"):
            hero = state.get_hero()
            if hero and hasattr(state, "players"):
                max_bet = max((p.current_bet_bb for p in state.players), default=0.0)
                to_call_bb = max(0.0, max_bet - hero.current_bet_bb)

        system_prompt = self.SMART_SYSTEM_PROMPT if mode == "SMART" else self.FAST_SYSTEM_PROMPT

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "format": "json",
            "keep_alive": "5m",
            "options": {
                "temperature": 0.0,
                "top_p": 0.9
            }
        }

        start_time = time.perf_counter()
        raw_response = ""

        try:
            response = requests.post(self.api_url, json=payload, timeout=10.0)
            latency_sec = time.perf_counter() - start_time

            if response.status_code == 200:
                raw_response = response.json().get("response", "")
                decision = self.parse_response(raw_response, to_call_bb)
            else:
                decision = EngineDecision(action="FOLD" if to_call_bb > 0 else "CHECK", reasoning="HTTP Error")
        except Exception as e:
            latency_sec = time.perf_counter() - start_time
            decision = EngineDecision(action="FOLD" if to_call_bb > 0 else "CHECK", reasoning=f"Exception: {e}")

        # Безопасное формирование данных для лога (учитывает отсутствие state)
        hero_cards = "None"
        board_cards = "None"
        hero_pos = "UNKNOWN"
        pot_bb = 0.0

        if state:
            if hasattr(state, "hero_cards") and state.hero_cards:
                hero_cards = " ".join([c.to_pokerbench() for c in state.hero_cards])
            if hasattr(state, "board_cards") and state.board_cards:
                board_cards = " ".join([c.to_pokerbench() for c in state.board_cards])
            if hasattr(state, "hero_position") and state.hero_position:
                hero_pos = state.hero_position.value
            if hasattr(state, "pot_bb"):
                pot_bb = state.pot_bb

        state_summary = {
            "hero_position": hero_pos,
            "hero_cards": hero_cards,
            "board": board_cards,
            "pot_bb": pot_bb,
            "to_call_bb": to_call_bb
        }

        # Запись в логер
        self.logger.log_decision(
            state_summary=state_summary,
            prompt=prompt,
            raw_response=raw_response,
            parsed_action=decision.action,
            parsed_amount=decision.amount_bb,
            reasoning=decision.reasoning,
            latency_sec=latency_sec,
            mode=mode
        )

        return decision