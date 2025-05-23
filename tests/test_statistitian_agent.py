import pytest
from app.agents.statistician_agent import StatisticianAgent
from tests.LLM_evaluator import LlmEvaluator
import duckdb
import pandas as pd

@pytest.fixture
def test_StatisticianAgent():
    return StatisticianAgent()


@pytest.mark.parametrize("query,expected_matches", [
    ("SELECT * FROM table;", 0),
    ("DELETE FROM table;", 1),
    ("UPDATE users SET name = 'x';", 1),
    ("CREATE TABLE new_table (...);", 1),
    ("SELECT * FROM users; DELETE FROM users;", 1),
])
def test_check_query_regex(test_StatisticianAgent, query, expected_matches):
    agent = test_StatisticianAgent
    assert agent.check_query_regex(query) == expected_matches


@pytest.mark.parametrize(
    "question,expected_response",
    [
        ("What is the highest ranked product","To answer this question you need to group by type of product (class_name) and average rating")
    ]
)
def test_prepare_stat_analysis(test_StatisticianAgent,question,expected_response):
    agent = test_StatisticianAgent
    llm_response = agent.prepare_stat_analysis(question)
    print(llm_response)
    llm_evaluator = LlmEvaluator()
    scores = llm_evaluator.evaluate(question,expected_response,llm_response)
    print(scores)
    test_passed = True
    for metric, score in scores.items():
        if score < 50:
            test_passed = False

    assert test_passed == True


@pytest.mark.parametrize(
    "question, prev_step_answer, expected_response",
    [
        ("What type of product has the most extreme reviews",
         "To answeet the question you have to group products by type class_name and then calculate standard deviation of the responses, return the results sorted by the highest value of calculated std",
         "SELECT class_name, STDDEV_POP(rating) FROM reviews GROUP BY class_name ORDER BY STDDEV_POP(rating) DESC")
    ]
)
def test_prepare_sql_query(test_StatisticianAgent,question,prev_step_answer,expected_response):
    agent = test_StatisticianAgent
    llm_response = agent.prepare_sql_query(question, prev_step_answer)
    print(llm_response)
    llm_evaluator = LlmEvaluator()
    scores = llm_evaluator.evaluate(question,expected_response,llm_response)
    print(scores)
    test_passed = True
    for metric, score in scores.items():
        if score < 10:
            test_passed = False

    assert test_passed == True


@pytest.mark.parametrize(
    "query, expected_response",
    [
    ("SELECT user FROM table WHERE user = 1;", True),
    ("DELETE FROM table;", False),
    ("SELECT SELECT user;", False),
    ("SELECT class_name, STDDEV_POP(rating) FROM reviews GROUP BY class_name ORDER BY STDDEV_POP(rating) DESC;", True)
    ]
)
def test_check_query_llm(test_StatisticianAgent,query,expected_response):
    agent = test_StatisticianAgent
    llm_response = agent.check_query_llm(query)
    if "YES" in llm_response:
        safe = True
    else:
        safe = False
    assert safe == expected_response


@pytest.mark.parametrize(
    "query, expected_response",
    [
    ("SELECT class_name, STDDEV_POP(rating) FROM reviews GROUP BY class_name ORDER BY STDDEV_POP(rating) DESC", "stddev_pop(rating)"),
    ("""
    SELECT class_name, AVG(rating) AS avg_rating, STDDEV_POP(rating) AS stddev_rating
    FROM reviews
    GROUP BY class_name
    ORDER BY STDDEV_POP(rating) DESC
    LIMIT 5;
     """, "avg_rating")
    ]
)
def test_execute_query(query,expected_response):
    with duckdb.connect("app/database/reviews.duckdb") as con:
        result = con.execute(query).fetchdf().to_string()
    assert expected_response in result



def test_run_happy_path(mocker):
    agent = StatisticianAgent()
    mocker.patch.object(agent, "prepare_stat_analysis", return_value="Instruction")
    sql_output = "```sql SELECT DISTINCT(class_name) FROM reviews; ```"
    mocker.patch.object(agent, "prepare_sql_query", return_value=sql_output)
    mocker.patch.object(agent, "check_query_regex", return_value=0)
    mocker.patch.object(agent, "check_query_llm", return_value="YES, it's safe")
    dummy_df = pd.DataFrame({'class_name': {0: 'Outerwear',
                                            1: 'Sweaters',
                                            2: 'Intimates',
                                            3: 'Lounge',
                                            4: 'Knits',
                                            5: 'Dresses',
                                            6: 'Blouses',
                                            7: 'Skirts',
                                            8: 'Sleep',
                                            9: 'Fine gauge',
                                            10: 'Pants'}})
    mocker.patch.object(agent, "execute_query", return_value=dummy_df.to_string())
    result = agent.run("Get all types of products (class name)")
    assert "Outerwear" in result



def test_run_unsafe_query(mocker):
    agent = StatisticianAgent()
    mocker.patch.object(agent, "prepare_stat_analysis", return_value="Instruction")
    mocker.patch.object(agent, "prepare_sql_query", return_value="```sql DELETE FROM users; ```")
    mocker.patch.object(agent, "check_query_regex", return_value=1)
    result = agent.run("Delete all users")
    assert "Table modifications are not allowed" in result

