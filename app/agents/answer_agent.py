from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages.ai import AIMessage


class AnswerAgent():
    def __init__(self, memory):
        self.memory = memory
        llm = ChatOllama(model="mistral:7b", 
                 temperature=0)
    
        prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to answer the Question based only on the provided Information Context.
            Make it brief and informative for the store manager, try to obtain business insights.

            Information Context: {information_context}.

            Question: {question}
            """
        )

        self.main_decision_chain = prompt | llm

    def invoke(self, information_context: str, question: str) -> bool:
        print("Final answer agent")
        result = self.main_decision_chain.invoke({"information_context": information_context, "question": question})
        self.memory.chat_memory.add_message(AIMessage(result.content))
        return result.content