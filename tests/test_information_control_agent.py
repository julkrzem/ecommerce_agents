import pytest
from agents.information_control_agent import ContextAssesmentAgent


@pytest.fixture
def test_ContextAssesmentAgent():
    return ContextAssesmentAgent()


@pytest.mark.parametrize(
    "context, question , expected",
    [
        ("","What are customers saying about dresses?", False),
        ("[]", "What are customers saying about dresses?", False),
        ("I like coffe", "What are customers saying about dresses?", False),
        ("I like the dress", "What are customers saying about the website design?", False),
        ("Customer 1: Dress is so pretty, Customer 2: I love it, Customer 3: Nice dress", "What are customers saying about dresses?",True),
        ("Customer 1: I love the website design! Customer 2: The icons shold be bigger","What are customers saying about the website design?", True)
    ]
)
def test_context_sufficient(test_ContextAssesmentAgent, context, question, expected):
    agent = test_ContextAssesmentAgent
    assert agent.context_sufficient(context, question) == expected