"""MAS divergence: stuck-triggered hard action override.

When the checker blocks an agent's action because the agent is detected to be
in a stuck loop (via the dHash no_change trigger), this module produces K=3
candidate alternative actions from agents with different views of history,
filters out candidates matching recently-executed action signatures, and
selects one for hard execution by the runner.

Design (see design doc 2026-05-24):
- 3 candidate agents with different history modes: warm, cold, skip_recent
- Each gets an anti-bias text block listing recently-ineffective actions
- Filter rejects candidates whose action signature matches recent_sigs
- Selection from valid candidates: random pick
- Fallback (F1): all candidates filtered → execute navigate_back
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

# Re-use action type constant from existing models
from mobile_world.runtime.utils.models import JSONAction


# --------------------------------------------------------------------------
# Action signature: coarse-grained identity used for filter matching
# --------------------------------------------------------------------------


def action_signature(
    action: dict | None,
    click_tol: int = 80,
    drag_tol: int = 100,
) -> tuple:
    """Reduce an action dict to a coarse signature for repetition detection.

    The signature is bucketed by coordinate cluster so near-identical clicks
    or drags collapse to the same key.
    """
    if not action:
        return ("none",)
    t = action.get("action_type", "")
    if t == "click":
        return (
            "click",
            (action.get("x", 0) // click_tol) * click_tol,
            (action.get("y", 0) // click_tol) * click_tol,
        )
    elif t == "long_press":
        return (
            "long_press",
            (action.get("x", 0) // click_tol) * click_tol,
            (action.get("y", 0) // click_tol) * click_tol,
        )
    elif t == "drag":
        return (
            "drag",
            (action.get("start_x", 0) // drag_tol) * drag_tol,
            (action.get("start_y", 0) // drag_tol) * drag_tol,
            (action.get("end_x", 0) // drag_tol) * drag_tol,
            (action.get("end_y", 0) // drag_tol) * drag_tol,
        )
    elif t == "input_text":
        return ("input_text", action.get("text", "")[:30])
    else:
        return (t,)


def compute_recent_sigs(
    executed_actions: list[dict],
    n: int = 5,
    click_tol: int = 80,
    drag_tol: int = 100,
) -> set[tuple]:
    """Get the set of signatures from the last n executed actions."""
    return {
        action_signature(a, click_tol=click_tol, drag_tol=drag_tol)
        for a in executed_actions[-n:]
    }


def compute_sig_frequencies(
    executed_actions: list[dict],
    n: int = 5,
    click_tol: int = 80,
    drag_tol: int = 100,
) -> dict[tuple, int]:
    """Get frequency counts of action signatures in the last n executed actions."""
    sigs = [
        action_signature(a, click_tol=click_tol, drag_tol=drag_tol)
        for a in executed_actions[-n:]
    ]
    return dict(Counter(sigs))


# --------------------------------------------------------------------------
# Anti-bias text block
# --------------------------------------------------------------------------


def format_anti_bias_text(
    recent_sigs: set[tuple],
    freq_counter: dict[tuple, int] | None = None,
) -> str:
    """Build a human-readable constraint block listing ineffective actions."""
    if not recent_sigs:
        return ""
    freq_counter = freq_counter or {}
    lines = []
    for sig in recent_sigs:
        count = freq_counter.get(sig, 1)
        if sig[0] == "click":
            lines.append(
                f"  ・ click in region ({sig[1]}, {sig[2]}) — "
                f"attempted {count} time(s), no visible effect"
            )
        elif sig[0] == "long_press":
            lines.append(
                f"  ・ long_press in region ({sig[1]}, {sig[2]}) — "
                f"attempted {count} time(s), no visible effect"
            )
        elif sig[0] == "drag":
            lines.append(
                f"  ・ drag from ({sig[1]}, {sig[2]}) to ({sig[3]}, {sig[4]}) — "
                f"attempted {count} time(s), no visible effect"
            )
        elif sig[0] == "input_text":
            lines.append(
                f"  ・ input_text \"{sig[1]}\" — "
                f"attempted {count} time(s), no visible effect"
            )
        else:
            lines.append(
                f"  ・ {sig[0]} — attempted {count} time(s), no visible effect"
            )
    return (
        "⚠️ STRICT CONSTRAINT — The following recent actions produced no visible "
        "effect on the screen:\n\n"
        + "\n".join(lines)
        + "\n\nThese actions are confirmed to NOT work. Do NOT propose any of them again. "
        + "Propose a substantively different action (different coordinate region, "
        + "different action type, or different UI target)."
    )


# --------------------------------------------------------------------------
# Fallback action (F1)
# --------------------------------------------------------------------------


def fallback_action() -> dict:
    """The F1 fallback: navigate back. Used when all candidates are filtered."""
    return {"action_type": "navigate_back"}


# --------------------------------------------------------------------------
# Result dataclasses
# --------------------------------------------------------------------------


@dataclass
class CandidateResult:
    agent_role: str  # "warm" | "cold" | "skip_recent"
    action: dict | None
    prediction_text: str | None
    filtered: bool = False
    filter_reason: str = ""
    parse_failed: bool = False

    def model_dump(self) -> dict:
        return asdict(self)


@dataclass
class MASDivergenceResult:
    engaged: bool
    candidates: list[CandidateResult]
    selected_role: str  # "warm" | "cold" | "skip_recent" | "fallback"
    selected_action: dict
    fallback_used: bool
    recent_sigs: list[tuple]

    def model_dump(self) -> dict:
        return {
            "engaged": self.engaged,
            "candidates": [c.model_dump() for c in self.candidates],
            "selected_role": self.selected_role,
            "selected_action": self.selected_action,
            "fallback_used": self.fallback_used,
            "recent_sigs": [list(s) for s in self.recent_sigs],
        }


# --------------------------------------------------------------------------
# Main entry: run_mas_divergence
# --------------------------------------------------------------------------


def run_mas_divergence(
    agent: Any,
    observation: dict,
    task_goal: str,
    executed_actions: list[dict],
    skip_recent_k: int = 3,
    recent_sig_window: int = 5,
    click_tol: int = 80,
    drag_tol: int = 100,
    rejected_action: dict | None = None,
) -> MASDivergenceResult:
    """Run MAS divergence: generate 3 candidates, filter, random pick from valid.

    Args:
        agent: the main agent instance (must expose propose_candidate)
        observation: current step observation (screenshot etc.)
        task_goal: the task goal text
        executed_actions: list of actions that were actually executed (not blocked).
            Most recent at the end.
        skip_recent_k: for "skip_recent" agent, drop the last K steps of history
        recent_sig_window: recent_sigs is computed from the last N executed actions
        click_tol/drag_tol: coord clustering tolerance for signature matching
        rejected_action: the action the checker just blocked. If provided, its
            signature is added to recent_sigs so candidates and anti-bias text
            include it. Critical for completion_check_loop where the rejected
            FINISHED/ANSWER action is NOT in executed_actions.

    Returns:
        MASDivergenceResult with all candidates' metadata and the selected action.
    """
    recent_sigs = compute_recent_sigs(
        executed_actions, n=recent_sig_window, click_tol=click_tol, drag_tol=drag_tol
    )
    freq_counter = compute_sig_frequencies(
        executed_actions, n=recent_sig_window, click_tol=click_tol, drag_tol=drag_tol
    )

    # If a specific rejected action is given (e.g. finished/answer blocked by
    # completion_check_loop), merge its signature in so candidates avoid it.
    if rejected_action is not None:
        rej_sig = action_signature(
            rejected_action, click_tol=click_tol, drag_tol=drag_tol
        )
        recent_sigs.add(rej_sig)
        # Treat the rejected action as "attempted N+1 times" for the anti-bias
        # text (N is the number of times it already appeared in executed_actions,
        # which is typically 0 for a blocked finished/answer).
        freq_counter[rej_sig] = freq_counter.get(rej_sig, 0) + 1

    anti_bias = format_anti_bias_text(recent_sigs, freq_counter)

    candidates: list[CandidateResult] = []
    for role in ("warm", "cold", "skip_recent"):
        try:
            pred_text, action_dict = agent.propose_candidate(
                observation=observation,
                task_goal=task_goal,
                history_mode=role,
                skip_recent_k=skip_recent_k,
                extra_instruction=anti_bias,
            )
            candidates.append(
                CandidateResult(
                    agent_role=role,
                    action=action_dict,
                    prediction_text=pred_text,
                )
            )
        except Exception as exc:
            logger.warning(
                "MAS candidate {} failed: {}",
                role,
                exc,
            )
            candidates.append(
                CandidateResult(
                    agent_role=role,
                    action=None,
                    prediction_text=None,
                    parse_failed=True,
                )
            )

    # Filter
    for c in candidates:
        if c.parse_failed:
            c.filtered = True
            c.filter_reason = "parse_failed"
            continue
        sig = action_signature(c.action, click_tol=click_tol, drag_tol=drag_tol)
        if sig in recent_sigs:
            c.filtered = True
            c.filter_reason = f"signature_in_recent: {sig}"

    valid = [c for c in candidates if not c.filtered]

    if valid:
        chosen = random.choice(valid)
        return MASDivergenceResult(
            engaged=True,
            candidates=candidates,
            selected_role=chosen.agent_role,
            selected_action=chosen.action,
            fallback_used=False,
            recent_sigs=list(recent_sigs),
        )
    else:
        # F1 fallback
        logger.warning(
            "All MAS candidates filtered; using F1 fallback (navigate_back). "
            "recent_sigs={}",
            list(recent_sigs),
        )
        return MASDivergenceResult(
            engaged=True,
            candidates=candidates,
            selected_role="fallback",
            selected_action=fallback_action(),
            fallback_used=True,
            recent_sigs=list(recent_sigs),
        )
