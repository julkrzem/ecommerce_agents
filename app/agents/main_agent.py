from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from app.agents.answer_agent import AnswerAgent
from app.agents.rag_agent import RagAgent
from app.agents.statistician_agent import StatisticianAgent
from langchain.output_parsers.json import SimpleJsonOutputParser


class MainAgentSupervisor:
    """
    A class representing agent responsible for main decisions and supervising other sub-agents and data collection process

    Attributes
    ----------
    llm : ChatOllama
        Instance of LLM
    context_assesment_prompt : PromptTemplate
        Template of prompt to check if context is sufficient to answer the question
    agent_selection_prompt: PromptTemplate
        Template of prompt to select a sub-agent for context collection
    json_parser: SimpleJsonOutputParser()
        Json parser for LangChain chain
    context_assesment_chain: Chain
        LangChain chain for context assesment
    agent_selection_chain: Chain
        LangChain chain for agent selection
    answer_llm: Answer Agent
        Instance of AnswerAgent
    rag_agent: RagAgent
        Instance of Rag Agent
    statistician_agent: StatisticianAgent
        Instance of Statistician Agent

    Methods
    -------
    context_assesment(question: str)
        Running LLM decision if collected context is sufficient to answer user question
    invoke(question: str, history: str = None)
        Executes the Main agentic workflow
    """
    def __init__(self):
        llm = ChatOllama(model="mistral:7b", 
                        temperature=0,
                        base_url = "http://host.docker.internal:11434")
        
        context_assesment_prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to decide if there is enough information in provided Information Context to answer user question. The decision must be based on the Information Context. 
            If the Information Context is empty: answer NO.
            If the Information Context does not have key information to answer directly the user question: answer NO. 
            If the Information Context provides necessary information to answer directly the user question: answer YES.

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
        self.answer_llm = AnswerAgent()
        self.rag_agent = RagAgent()
        self.statistician_agent = StatisticianAgent()

    def context_assesment(self, question: str, collected_context: str) -> str:
        response = self.context_assesment_chain.invoke({"information_context": collected_context, "question": question}).content.upper()
        return response


    def invoke(self, question: str, history: str = None) -> str:
        """
        Runs the Main agent workflow

        Parameters
        ----------
        question: str
            User input question
        history: str, optional
            Chat history from previous messages

        Returns
        -------
        str
            Collected knowledge - text based on the history and sub-agents answers
        """
        if history:
            collected_context = history
        else: collected_context = ""
        
        iteration = 0
        max_iteration = 3

        while iteration < max_iteration:
            iteration += 1

            if "YES" in self.context_assesment(question, collected_context):
                return self.answer_llm.invoke(collected_context,question)
            else:
                if len(collected_context)>=4000:
                    collected_context = collected_context[-4000:]

                agent = self.agent_selection_chain.invoke({"question":question})

                if agent["agent"] == "rag_agent":
                    print("RAG Agent used")
                    retrived_content = self.rag_agent.run(question)
                    collected_context += retrived_content
                    

                if agent["agent"] == "statistician_agent":
                    print("STAT Agent used")
                    agent = self.statistician_agent
                    retrived_content = agent.run(question)
                    collected_context += retrived_content
            
            print("--------------------")
            print(collected_context)
            print("--------------------")
        
        collected_context = "Max iterations exceeded - There is not enough context to answer user question!"
        return self.answer_llm.invoke(collected_context,question)