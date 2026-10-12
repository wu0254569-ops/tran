"""One round of the core loop: decision -> environment step -> reflection -> memory.

The interface (RoundInput / DecisionOutput / EnvironmentStep / ReflectionOutput)
is stable; the loop implementation can be replaced by a new framework later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.io.parse import as_bool, extract_json, find_action
from src.loop.schema import LongTermOp, Observation

RULES = (
    "You are playing the game 2048 on a 4x4 grid. Tiles slide when a direction is "
    "chosen; equal tiles merge and increase the score. Reach 2048. Valid moves are: "
    "up, down, left, right."
)

DECISION_TEMPLATE = (
    "{rules}\n\n"
    "{state}\n\n"
    "Short-term memory (your cache):\n{short_memory}\n\n"
    "Long-term memory index (keys only): {long_index}\n"
    "{long_context}\n"
    "Think briefly, then choose exactly one move.\n"
    'Reply with JSON only: {{"self_thought": "...", "self_action": "up|down|left|right"}}'
)

REFLECTION_TEMPLATE = (
    "Before the move:\n{state_before}\n\n"
    "You chose: {action}\n\n"
    "After the move:\n{state_after}\n\n"
    "Judge whether the move was good or bad and reflect. If you think this lesson "
    "is developable, request fine-tuning; simple lessons should go to long-term "
    "memory instead.\n"
    "Reply with JSON only: {{\"result_evaluation\": \"...\", \"self_reflection\": \"...\", "
    "\"wrong_action\": \"...\", \"right_action\": \"...\", \"finetune\": true, "
    "\"long_term_ops\": [], \"short_term_memory\": \"...\"}}"
)


@dataclass
class RoundReport:
    round_index: int
    state_before: str
    action: str
    self_thought: str
    state_after: str
    score: int
    over: bool
    evaluation: str = ""
    self_reflection: str = ""
    wrong_action: str = ""
    right_action: str = ""
    finetune: bool = False
    train_metrics: dict[str, float] | None = None
    short_term_memory: str = ""
    notes: list[str] = field(default_factory=list)


class RoundRunner:
    def __init__(
        self,
        model,
        memory,
        actions: list[str],
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.memory = memory
        self.actions = list(actions)
        self.max_new_tokens = max_new_tokens
        self._long_context = ""

    # -- public ------------------------------------------------------------
    def run(self, env, observation: Observation, round_index: int) -> tuple[RoundReport, Observation]:
        short_memory = self.memory.read_short()
        decision = self._decide(observation, short_memory, round_index)
        report = RoundReport(
            round_index=round_index,
            state_before=observation.text,
            action=decision["action"],
            self_thought=decision["thought"],
            state_after="",
            score=observation.score,
            over=observation.over,
        )

        result, done, info = env.step(decision["action"])
        report.state_after = result.text
        report.score = info["score"]
        report.over = info["over"]

        reflection = self._reflect(observation, result, decision["action"], short_memory)
        report.evaluation = reflection["evaluation"]
        report.self_reflection = reflection["self_reflection"]
        report.wrong_action = reflection["wrong_action"]
        report.right_action = reflection["right_action"]
        report.finetune = reflection["finetune"]

        if reflection["finetune"]:
            if reflection["right_action"] in self.actions:
                report.train_metrics = self.model.train_step(
                    decision["input_ids"],
                    decision["attention_mask"],
                    self.model.action_token_id(reflection["right_action"]),
                )
            else:
                report.notes.append("skipped fine-tune: right_action not in actions")

        if reflection["long_term_ops"]:
            self.memory.apply_ops(reflection["long_term_ops"])
        self.memory.write_short(reflection["short_term_memory"])
        report.short_term_memory = reflection["short_term_memory"]

        self._refresh_long_context(reflection["long_term_read"])
        return report, result

    # -- stages ------------------------------------------------------------
    def _decide(self, observation: Observation, short_memory: str, round_index: int) -> dict:
        rules = RULES if round_index == 0 else "2048: moves are up/down/left/right."
        prompt = DECISION_TEMPLATE.format(
            rules=rules,
            state=observation.text,
            short_memory=short_memory or "(empty)",
            long_index=self.memory.long_index(),
            long_context=self._long_context,
        )
        input_ids, attention_mask = self.model.encode_messages(
            [{"role": "user", "content": prompt}]
        )
        raw = self.model.generate_ids(input_ids, attention_mask, self.max_new_tokens)
        parsed = extract_json(raw) or {}
        action = str(parsed.get("self_action", "")).strip().lower()
        if action not in self.actions:
            action = find_action(raw, self.actions) or self.actions[0]
        thought = str(parsed.get("self_thought", raw)).strip()
        long_term_read = parsed.get("long_term_read", [])
        if not isinstance(long_term_read, list):
            long_term_read = []
        return {
            "action": action,
            "thought": thought,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "raw": raw,
            "long_term_read": [str(k) for k in long_term_read],
        }

    def _reflect(
        self,
        before: Observation,
        after: Observation,
        action: str,
        short_memory: str,
    ) -> dict:
        prompt = REFLECTION_TEMPLATE.format(
            state_before=before.text,
            action=action,
            state_after=after.text,
        )
        input_ids, attention_mask = self.model.encode_messages(
            [{"role": "user", "content": prompt}]
        )
        raw = self.model.generate_ids(input_ids, attention_mask, self.max_new_tokens)
        parsed = extract_json(raw) or {}

        right_action = str(parsed.get("right_action", "")).strip().lower()
        if right_action not in self.actions:
            right_action = find_action(str(parsed.get("self_reflection", "")) + " " + raw,
                                       self.actions) or ""
        wrong_action = str(parsed.get("wrong_action", action)).strip().lower() or action

        ops: list[LongTermOp] = []
        for item in parsed.get("long_term_ops", []) or []:
            if isinstance(item, dict) and item.get("key"):
                ops.append(
                    LongTermOp(op=str(item.get("op", "write")), key=str(item["key"]),
                               value=item.get("value"))
                )

        memory = parsed.get("short_term_memory", "")
        if isinstance(memory, (list, tuple)):
            memory = "\n".join(str(x) for x in memory)

        return {
            "evaluation": str(parsed.get("result_evaluation", "")).strip(),
            "self_reflection": str(parsed.get("self_reflection", raw)).strip(),
            "wrong_action": wrong_action,
            "right_action": right_action,
            "finetune": as_bool(parsed.get("finetune", False)),
            "long_term_ops": ops,
            "short_term_memory": str(memory).strip(),
            "long_term_read": [],
        }

    def _refresh_long_context(self, keys: list[str]) -> None:
        if not keys:
            return
        entries = self.memory.read_long([str(k) for k in keys])
        if entries:
            self._long_context = "Long-term memory:\n" + "\n".join(
                f"- {k}: {v}" for k, v in entries.items()
            )
