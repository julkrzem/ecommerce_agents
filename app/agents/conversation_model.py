from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate

from app.agents.main_agent import MainAgentSupervisor


class Chat:
    def __init__(self):

        self.classifier_prompt = PromptTemplate.from_template(
            """You are a e-commerce store assistant. Your job is to decide if the following user message requires looking into data from the store database or into company data, or it is only general conversation.
            User question: {question}

            Any question regarding the quality, quantity, statistics, score, review content is "company_data" 

            If it is general question answer "simple" simple question means: hello, basic conversation or general question  e.g. "what does a department mean?"

            If it is a question based on company data answer "company_data"
            
            Answer with only "simple" or "company_data".
            """)

        self.prompt = PromptTemplate.from_template(
            """
            You are a friendly chatbot that answers simply to the user questions. Do not write anything unnecessary, just short answers or follow-up questions to user messages. Keep responses short.

            Previous conversation:
            {history}

            User: {input}
            Assistant:"""
            )

        self.llm = ChatOllama(model="mistral:7b", 
                        temperature=0.7)
        
    def create_chain(self, prompt):
        chain = prompt | self.llm
        return chain

    def use_agent(self, user_input: str) -> bool:
        chain = self.create_chain(self.classifier_prompt)
        result = chain.invoke(user_input).content.lower()
        return "company_data" in result

    def run(self, user_input: str, history: list) -> str:

        history = str(history)

        if len(history)>=1000:
            history = history[-1000:]
        
        if self.use_agent(user_input):
            print("complex")
            agent = MainAgentSupervisor()
            response = agent.invoke(user_input, history)
        else:
            print("simple")
            chain = self.create_chain(self.prompt)
            response = chain.invoke({"input": user_input, "history": history}).content

        return response