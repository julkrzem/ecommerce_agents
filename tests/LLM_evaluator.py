from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser

class LlmEvaluator:
    def __init__(self):
        llm = ChatOllama(model="mistral:7b", 
                 temperature=0)
    
        evaluator_prompt = """You are a LLM model evaluator that is responsible for assesment of other LLM's responses. 
            You will be given a user question and expected response, as well as LLM model response. 
            The task is to score the response, in range [0-100] where 0 is the lowest score, based on the criteria:

            1. accuracy: Is the answer correct comparing to the example output? [0-100]
            2. relevance: Is the answer directly related to the user's query, and avoids unnecessary or off-topic information [0-100]
            3. clarity: Is the answer clear and easy to understand? [0-100]
            4. correctness: Is the answer grammatically correct? [0-100]

            User question: {question}
            
            Note that the expected response is usually simplified, so check if the facts are correct.
            Expected response: {expected_response}


            Answer with JSON with keys accuracy,relevance,clarity,correctness and their corresponding score."""
        
        user_message = "LLM model response: {llm_response}"

        sql_prompt_template = ChatPromptTemplate(
            [("system", evaluator_prompt), ("user", user_message)]
        )

        json_parser = JsonOutputParser()

        self.eval_chain = sql_prompt_template | llm |json_parser

    def evaluate(self, question,expected_response,llm_response):
        response = self.eval_chain.invoke({"question":question,"expected_response":expected_response,"llm_response":llm_response})
        return response
        
