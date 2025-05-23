from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate

from main_agent import MainAgentSupervisor

class Chat:
    def __init__(self, memory):
        self.memory = memory

        classifier_prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to decide if the following user message requires external data source or specific tool action to answer their question.
            User question: {question}
            Answer with only "simple" or "agent".
            """)

        system_prompt = """You are a friendly chatbot that answers simply to the user questions. Do not write anything unnecessary, just short answers or follow-up questions to user messages. Keep responses short"""

        prompt = PromptTemplate.from_template(
            """
            {system_prompt}

            Previous conversation:
            {history}

            User: {input}
            Assistant:"""
            )

        llm = ChatOllama(model="mistral:7b", 
                        temperature=0.7)
        self.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt.partial(system_prompt=system_prompt), 
            verbose=True)

        self.classifier_chain = classifier_prompt | llm

    def use_agent(self, user_input: str) -> bool:
        result = self.classifier_chain.invoke(user_input).content.lower()
        return "agent" in result


    def run(self):
        print("Start chatting (type 'bye!')\n")

        while True:
            print(self.memory)
            user_input = input("What do you want to ask me?: ")
            if user_input.lower() in {"bye!"}:
                break

            if self.use_agent(user_input):
                print("complex")
                agent = MainAgentSupervisor(self.memory)
                response = agent.invoke(user_input)

            else:
                print("simple")
                response = self.conversation.invoke({"input":user_input})
                response = response["response"]

            print(response)



memory = ConversationBufferMemory()
conversation_chat = Chat(memory)
conversation_chat.run()