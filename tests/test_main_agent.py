import pytest
from app.agents.main_agent import MainAgentSupervisor
from tests.LLM_evaluator import LlmEvaluator

@pytest.fixture
def test_MainAgentSupervisor():
    return MainAgentSupervisor()


@pytest.mark.parametrize(
    "context, question , expected",
    [
        ("","What are customers saying about dresses?", False),
        ("[]", "What are customers saying about dresses?", False),
        ("Customer 1: Dress is so pretty, Customer 2: I love it, Customer 3: Nice dress", "What are customers saying about dresses?",True),
        ("Customer 1: I love the website design! Customer 2: The icons shold be bigger","What are customers saying about the website design?", True)
    ]
)
def test_context_assesment(test_MainAgentSupervisor, context, question, expected):
    agent = test_MainAgentSupervisor
    response = "YES" in agent.context_assesment(context, question)
    assert response == expected