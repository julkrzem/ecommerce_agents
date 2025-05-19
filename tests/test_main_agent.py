import pytest
from agents.main_agent import MainAgentSupervisor

@pytest.fixture
def test_MainAgentSupervisor():
    return MainAgentSupervisor()


@pytest.mark.parametrize(
    "question , expected",
    [
        ("What are customers saying about dresses?", "answer based on context")
    ]
)

def test_invoke(test_MainAgentSupervisor, question, expected):
    agent = test_MainAgentSupervisor
    assert agent.invoke(question) == expected