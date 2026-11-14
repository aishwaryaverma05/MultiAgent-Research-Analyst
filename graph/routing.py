
def route_after_critic(
    score: float,
    revision_count: int,
    max_revisions: int = 3,
) -> str:
    """
    Decide whether a report should be revised or sent for human review.

    Maximum revision attempts are strictly bounded.
    """

    if revision_count >= max_revisions:
        return "human_review"

    if score >= 7:
        return "human_review"

    return "revise"
