"""What the environment tells an agent after a submission.

Only whether it was wrong (optionally with the kind of error). Never the
tests, the expected outputs, how many tests passed, or the right answer.
"""

from __future__ import annotations

from src.config import FeedbackLevel
from src.models.code import ExecutionResult


def describe_failure(result: ExecutionResult, level: FeedbackLevel) -> str:
    """Feedback text for a submission ('' if it was accepted)."""
    if result.passed:
        return ""
    if level is FeedbackLevel.ERROR_TYPE:
        return (
            f"Your solution is wrong. Errors observed: {', '.join(result.error_types)}."
        )
    return "Your solution is wrong."
