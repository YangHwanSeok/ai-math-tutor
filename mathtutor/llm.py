import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

JsonCompleter = Callable[[list[dict]], dict]

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


def json_completer(model: str, instructions: str, schema: dict, name: str,
                   effort: str | None = None, usage: Usage | None = None) -> JsonCompleter:
    """대화 메시지 목록을 받아 schema에 맞는 JSON(dict)을 돌려주는 함수를 만든다."""
    def complete(messages: list[dict]) -> dict:
        response = _get_client().responses.create(
            model=model,
            instructions=instructions,
            input=messages,
            text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
            **({"reasoning": {"effort": effort}} if effort else {}),
        )
        if usage is not None:
            usage.calls += 1
            usage.input_tokens += response.usage.input_tokens
            usage.output_tokens += response.usage.output_tokens
        return json.loads(response.output_text)

    return complete
