import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

class DecisionLogger:
    def __init__(self, log_filepath: str = "logs/poker_decisions.jsonl"):
        self.log_filepath = log_filepath
        # Создаем папку logs, если ее нет
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)

    def log_decision(
        self,
        state_summary: Dict[str, Any],
        prompt: str,
        raw_response: str,
        parsed_action: str,
        parsed_amount: Optional[float],
        reasoning: Optional[str],
        latency_sec: float,
        mode: str
    ):
        """Записывает полную информацию о принятом решении в .jsonl файл."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "latency_sec": round(latency_sec, 3),
            "table_state": state_summary,
            "prompt": prompt,
            "raw_response": raw_response,
            "decision": {
                "action": parsed_action,
                "amount_bb": parsed_amount,
                "reasoning": reasoning
            }
        }

        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ [Logger Error]: Не удалось записать лог: {e}")