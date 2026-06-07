from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class MetricsCollector:
    model_name: str
    values: dict[str, list[int]] = field(
        default_factory=lambda: {
            "cached_answer": [],
            "llm_answer": [],
            "tts_first_sentence": [],
            "first_answer_time": [],
        }
    )

    def record(self, metric_name: str, value_ms: int) -> None:
        if metric_name not in self.values:
            self.values[metric_name] = []
        self.values[metric_name].append(int(value_ms))
        self.values[metric_name] = self.values[metric_name][-200:]

    def snapshot(self) -> dict[str, object]:
        payload: dict[str, object] = {"model_name": self.model_name}
        for name, items in self.values.items():
            average = int(sum(items) / len(items)) if items else 0
            maximum = max(items) if items else 0
            payload[name] = {
                "values": list(items),
                "average": average,
                "max": maximum,
            }
        return payload
