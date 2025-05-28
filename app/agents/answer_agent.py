from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class AnswerAgent():
    """
    A class representing an agent that generates a final conclusion based on the provided context

    Attributes
    ----------
    llm : ChatOllama
        Instance of LLM
    prompt : PromptTemplate
        Template of structured prompt
    final_answer_chain : Chain 
        A processing chain

    Methods
    -------
    invoke(information_context: str, question: str)
        Executes the answer generation chain
    """
    def __init__(self):
        llm = ChatOllama(model="mistral:7b", 
                 temperature=0)
    
        prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to answer the Question based only on the provided Information Context.
            Make it brief and informative for the store manager, try to obtain business insights.

            Information Context: {information_context}.

            Question: {question}
            """
        )

        self.final_answer_chain = prompt | llm

    def invoke(self, information_context: str, question: str) -> bool:
        """
        Executes/runs the answer generation chain to produce final answer

        Parameters
        ----------
        information_context : str
            Information collected by other Agents in the workflow.
        question : str
            User input question

        Returns
        -------
        str
            Summary/conclusion of the collected knowledge
        """
        print("Final answer agent")
        result = self.final_answer_chain.invoke({"information_context": information_context, "question": question})
        return result.content