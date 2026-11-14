
from graph.routing import route_after_critic


def test_high_score_routes_to_human_review():

    assert route_after_critic(8.0, 0) == "human_review"


def test_low_score_routes_to_revision():

    assert route_after_critic(6.0, 0) == "revise"


def test_revision_limit_prevents_infinite_loop():

    assert route_after_critic(4.0, 3) == "human_review"
