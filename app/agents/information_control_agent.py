from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class ContextAssesmentAgent:
    def __init__(self):
        llm = ChatOllama(model="mistral:7b", 
                 temperature=0)
    
        prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to decide if there is enough information in provided Information Context to answer user question. The decission must be based on the Information Context. 
            If the Information Context is empty answer NO.
            If the Information Context is not sufficient answer NO. 
            If the Information Context provides necessairy information to answer user question answer YES.

            Information Context: {information_context}.

            Question: {question}

            Answer with YES or NO."""
        )

        self.main_decision_chain = prompt | llm

    def context_sufficient(self, information_context: str, question: str) -> bool:
        result = self.main_decision_chain.invoke({"information_context": information_context, "question": question})
        return "YES" in result.content.upper()