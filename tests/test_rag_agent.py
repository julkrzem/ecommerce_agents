import pytest
from app.agents.rag_agent import RagAgent
from tests.LLM_evaluator import LlmEvaluator
import json

def format_str(s):
    s = s.strip().lower().replace(" ","").replace("\n","")
    s = s.replace("'", '"')
    return s


@pytest.fixture
def test_RagAgent():
    return RagAgent()


@pytest.mark.parametrize(
    "question , expected",
    [
        ("What are customers saying about products?", "no_filter_needed"),
        ("Why are there problems with sizing?", "no_filter_needed"),
        ("What are customers saying about dresses?", ("department_name='dresses'", "class_name='dresses'", """{'department_name':{'$eq':'dresses'}}""","""{'class_name':{'$eq':'dresses'}}""","{'department_name':'dresses'}" , "{'class_name':'dresses'}"))
    ]
)

def test_needs_filtering(test_RagAgent, question, expected):
    agent = test_RagAgent
    response = format_str(agent.needs_filtering(question))
    assert response in expected


@pytest.mark.parametrize(
    "query_draft , expected",
    [
        ("department_name='dresses'", """{'department_name':{'$eq':'dresses'}}"""),
        ("class_name='dresses'", """{'class_name':{'$eq':'dresses'}}"""),
        ("""{"class_name":{"$eq":"dresses"}}""", """{"class_name":{"$eq":"dresses"}}"""),
        ("""{'class_name':'dresses'} and {'division_name':'general petite'}""","""{"$and":[{"class_name":{"$eq":"dresses"}},{"division_name":{"$eq":"generalpetite"}}]}""")
    ]
)
def test_correct_query(test_RagAgent, query_draft, expected):
    
    agent = test_RagAgent
    output_format = """
                Output message in this format if filter consists of one item:
                {"column_1": {"$eq": "Value_1"}}

                Output message in this format if filter consists of multiple items:
                {"$and": [
                    {"column_1": {"$eq": "Value_1"}},
                    {"column_2": {"$eq": "Value_2"}}
                    ]
                }
                """
    exp = json.loads(format_str(expected))
    response = json.loads(format_str(agent.correct_query(query_draft, output_format)))
    
    assert response == exp

@pytest.mark.parametrize(
    "question , db_filter",
    [
        ("What are customers saying about dresses?", "{'department_name':{'$eq':'dresses'}}")
    ]
)

def test_query_vectorstore_returned_results(test_RagAgent, question, db_filter):
    agent = test_RagAgent
    llm_response = agent.query_vectorstore(question, format_str(db_filter))


    assert len(llm_response) > 0



@pytest.mark.parametrize(
    "question , db_filter, expected_response",
    [
        ("What are customers saying about dresses?","""{'department_name':{'$eq':'dresses'}}""",""" Customers generally appreciate their purchases, finding customer reviews helpful. Many praise specific dresses for quality, flattering design, conservative length, and variety. However, some report fit issues (dresses run small, especially in the bust area) and quality concerns (poor fabrics, design flaws like netting getting stuck). A few customers have returned dresses due to fabric ripping. Overall, general consensus is positive but with some complaints about fit and quality.""" )

    ]
)
def test_query_vectorstore(test_RagAgent,question,db_filter,expected_response):
    agent = test_RagAgent

    llm_response = agent.query_vectorstore(question, format_str(db_filter))

    print(llm_response)
    llm_evaluator = LlmEvaluator()

    scores = llm_evaluator.evaluate(question,expected_response,llm_response)
    print(scores)
    test_passed = True
    for metric, score in scores.items():
        if score < 50:
            test_passed = False

    assert test_passed == True
