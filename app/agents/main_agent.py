from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from answer_agent import AnswerAgent
from rag_agent import RagAgent
from langchain.output_parsers.json import SimpleJsonOutputParser


class MainAgentSupervisor:
    def __init__(self):
        llm = ChatOllama(model="mistral:7b", 
                 temperature=0)
    
        context_assesment_prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to decide if there is enough information in provided Information Context to answer user question. The decission must be based on the Information Context. 
            If the Information Context is empty answer NO.
            If the Information Context is not sufficient answer NO. 
            If the Information Context provides necessairy information to answer user question answer YES.

            Information Context: {information_context}.

            Question: {question}

            Answer with YES or NO."""
        )

        agent_selection_prompt = PromptTemplate.from_template(
            """You are Agent supervisor. Define which Agent should be called in order to answer the user question.
            Question: {question}

            Agents to select from:
            "rag_agent": performs RAG on reviews in the database. Best to answer the questions about text content of the reviews.
            "statistical_agent": performs statistical analysis on the reviews database. Best to use when there is a quantity analysis needed.

            "Return a JSON object with an `agent` key, and the exact agent name.

            """
            
        )
        json_parser = SimpleJsonOutputParser()
        self.context_assesment_chain = context_assesment_prompt | llm
        self.agent_selection_chain = agent_selection_prompt | llm | json_parser
        self.answer_llm = AnswerAgent()
        self.rag_agent = RagAgent()

    def invoke(self, question: str) -> str:
        collected_context = ""
        iteration = 0
        max_iteration = 2

        while iteration < max_iteration:
            iteration += 1

            if "YES" in self.context_assesment_chain.invoke({"information_context": collected_context, "question": question}).content.upper():
                # return "answer based on info"
                return self.answer_llm.invoke(collected_context,question)
            else:
                agent = self.agent_selection_chain.invoke({"question":question})
                if agent["agent"] == "rag_agent":
                    retrived_content = self.rag_agent.run(question)
                    collected_context += retrived_content
                    print("RAG Agent used")

                if agent["agent"] == "statistical_agent":
                    return "STAT agent"
                
        return "max iterations exceeded"
    

agent = MainAgentSupervisor()
# print(agent.invoke("What they say about Dresses in the General division?"))
print(agent.invoke("What they say about Knits?"))
# print(agent.invoke("What is the most popular product?"))