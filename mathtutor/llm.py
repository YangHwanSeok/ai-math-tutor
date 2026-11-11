import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

JsonCompleter = Callable[[list[dict]], dict]
RETRIES = 4

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:  # 실험에서 여러 스레드가 동시에 호출해도 집계가 꼬이지 않게 한다
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def to_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def json_completer(model: str, instructions: str, schema: dict, name: str,
                   effort: str | None = None, usage: Usage | None = None) -> JsonCompleter:
    """대화 메시지 목록을 받아 schema에 맞는 JSON(dict)을 돌려주는 함수를 만든다."""
    def complete(messages: list[dict]) -> dict:
        for attempt in range(RETRIES + 1):
            try:
                response = _get_client().responses.create(
                    model=model,
                    instructions=instructions,
                    input=messages,
                    text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                    **({"reasoning": {"effort": effort}} if effort else {}),
                )
                break
            except RateLimitError:
                # 실험 중 동시 요청이 많을 때 잔액이 있어도 일시적으로 429(크레딧 없음)가 오는 경우가 있어 잠시 뒤 다시 시도한다
                if attempt == RETRIES:
                    raise
                time.sleep(10 * (attempt + 1))
        if usage is not None:
            usage.add(response.usage.input_tokens, response.usage.output_tokens)
        return json.loads(response.output_text)

    return complete
