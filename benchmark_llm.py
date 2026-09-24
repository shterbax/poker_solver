import time
import requests
import json

# Конфигурация
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-4b-pokerbench-grpo"

# Реальный тестовый промпт, аналогичный генерации из ActionTracker
TEST_PROMPT = """Game Type: CASH_6MAX
Stakes: $0.10/$0.25 (NL25)
Hero Hand: [Ah Kh]
Board: [Qh Jh 2c]
Pot Size: 12.5 BB
To Call: 0.0 BB
Position: BTN
Postflop History:
- Preflop: Hero raises to 2.5 BB, BB calls.
- Flop: Board [Qh Jh 2c]. BB checks.

Output JSON with GTO action."""

SYSTEM_PROMPT = "Output ONLY a raw JSON: {\"action\": \"CHECK\", \"amount_bb\": 0.0"


def run_benchmark():
    print(f"🚀 Запуск теста производительности для '{MODEL_NAME}'...\n")

    # Переменная messages теперь четко определена:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEST_PROMPT}
    ]

    payload = {
        "model": MODEL_NAME,
        "prompt": TEST_PROMPT,
        "system": SYSTEM_PROMPT,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,

        }
    }

    start_time = time.perf_counter()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        end_time = time.perf_counter()

        if response.status_code == 200:
            data = response.json()
            total_duration_sec = end_time - start_time

            # Извлекаем метрики из системного ответа Ollama (в наносекундах)
            eval_count = data.get("eval_count", 0)  # Количество сгенерированных токенов
            eval_duration_ns = data.get("eval_duration", 1)  # Время самой генерации
            prompt_eval_duration_ns = data.get("prompt_eval_duration", 1)  # Время обработки промпта

            tokens_per_second = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0
            prompt_eval_sec = prompt_eval_duration_ns / 1e9

            print(messages)

            print("📊 РЕЗУЛЬТАТЫ БЕНЧМАРКА:")
            print("=" * 40)
            print(f"⏱️ Общее время ответа (Total Latency): {total_duration_sec:.3f} сек")
            print(f"⚡ Время обработки промпта (Prompt Eval): {prompt_eval_sec:.3f} сек")
            print(f"🚀 Скорость генерации: {tokens_per_second:.2f} токен/сек")
            print(f"🔤 Всего сгенерировано токенов: {eval_count}")
            print("=" * 40)
            print("\n📩 Ответ модели:")
            print(data.get("response", "").strip())
        else:
            print(f"❌ Ошибка сервера Ollama: status_code {response.status_code}")
    except Exception as e:
        print(f"❌ Ошибка выполнения запроса: {e}")


if __name__ == "__main__":
    run_benchmark()