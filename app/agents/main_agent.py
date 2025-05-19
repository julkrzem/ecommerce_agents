from context_assesment_agent import ContextAssesmentAgent
from answer_agent import AnswerAgent
from vectorstore_retriver import VectorStore


class MainAgentSupervisor:
    def __init__(self):
        self.context_assesment_agent = ContextAssesmentAgent()
        self.answer_llm = AnswerAgent()
        self.vectorstore = VectorStore()
        self.retriver = self.vectorstore.retriever

    def invoke(self, question: str) -> str:
        collected_context = ""
        iteration = 0
        max_iteration = 2

        while iteration < max_iteration:
            iteration += 1
            if self.context_assesment_agent.context_sufficient(collected_context,question):
                return self.answer_llm.invoke(collected_context,question)
            else:
                information = self.retriver.invoke(question)
                for i in information:
                    # print(i.page_content)
                    collected_context += i.page_content
                
        return "max iterations exceeded"
    

agent = MainAgentSupervisor()
print(agent.invoke("What are they saying about the dress?"))