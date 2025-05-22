import pytest
from agents.rag_agent import RagAgent


@pytest.fixture
def test_RagAgent():
    return RagAgent()


@pytest.mark.parametrize(
    "question , expected",
    [
        ("What are customers saying about products?", "no_filter_needed"),
        ("Why are there problems with sizing?", "no_filter_needed"),

        # ("What they say about Dresses in the General division?", False),
        # ("What they say about Knits?", False),
    ]
)
def test_needs_filtering(test_RagAgent, question, expected):
    agent = test_RagAgent
    response = agent.needs_filtering(question)
    assert response == expected