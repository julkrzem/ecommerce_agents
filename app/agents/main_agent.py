from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from answer_agent import AnswerAgent
from rag_agent import RagAgent
from statistician_agent import StatisticianAgent
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.messages.ai import AIMessage
from langchain.memory import ConversationBufferMemory


class MainAgentSupervisor:
    def __init__(self, memory):
        self.memory = memory
        llm = ChatOllama(model="mistral:7b",
                         temperature=0)
        
        context_assesment_prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to decide if there is enough information in provided Information Context to answer user question. The decision must be based on the Information Context. 
            If the Information Context is empty answer NO.
            If the Information Context is not sufficient answer NO. 
            If the Information Context provides necessary information to answer user question answer YES.

            Information Context: {information_context}.

            Question: {question}

            Answer with YES or NO."""
        )

        agent_selection_prompt = PromptTemplate.from_template(
            """You are Agent supervisor. Define which Agent should be called in order to answer the user question.
            Question: {question}

            Agents to select from:
            "rag_agent": performs RAG on reviews in the database. Best to answer the questions about text content of the reviews.
            "statistician_agent": performs statistical analysis on the reviews database. Best to use when there is a quantity analysis needed.

            "Return a JSON object with an `agent` key, and the exact agent name.
            """
            
        )
        json_parser = SimpleJsonOutputParser()
        self.context_assesment_chain = context_assesment_prompt | llm
        self.agent_selection_chain = agent_selection_prompt | llm | json_parser
        self.answer_llm = AnswerAgent(self.memory)
        self.rag_agent = RagAgent(self.memory)
        self.statistician_agent = StatisticianAgent(self.memory)
        self.collected_context = ""

    def context_assesment(self, question: str) -> str:
        response = self.context_assesment_chain.invoke({"information_context": self.collected_context, "question": question}).content.upper()
        return response


    def invoke(self, question: str) -> str:

        for message in self.memory.chat_memory.messages:
            self.collected_context += "\n"+message.content

        if len(self.collected_context) >= 4000:
            self.collected_context = self.collected_context[-4000:]

        
        iteration = 0
        max_iteration = 3

        while iteration < max_iteration:
            iteration += 1

            if "YES" in self.context_assesment(question):
                return self.answer_llm.invoke(self.collected_context,question)
            else:
                agent = self.agent_selection_chain.invoke({"question":question})

                if agent["agent"] == "rag_agent":
                    print("RAG Agent used")
                    retrived_content = self.rag_agent.run(question)
                    self.collected_context += retrived_content
                    

                if agent["agent"] == "statistician_agent":
                    print("STAT Agent used")
                    agent = self.statistician_agent
                    retrived_content = agent.run(question)
                    self.collected_context += retrived_content
       
        return "max iterations exceeded"