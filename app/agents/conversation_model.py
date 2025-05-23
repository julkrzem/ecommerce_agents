from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate

from main_agent import MainAgentSupervisor

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

user_message = "Question: {question}"

memory = ConversationBufferMemory()

llm = ChatOllama(model="mistral:7b", 
                 temperature=0.7)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt.partial(system_prompt=system_prompt), 
    verbose=True)

classifier_chain = classifier_prompt | llm

def use_agent(user_input: str) -> bool:
    result = classifier_chain.invoke(user_input).content.lower()
    return "agent" in result

print("Start chatting (type 'bye!')\n")

while True:
    user_input = input("What do you want to ask me?: ")
    if user_input.lower() in {"bye!"}:
        break

    if use_agent(user_input):
        print("complex")
        agent = MainAgentSupervisor()
        response = agent.invoke(user_input)

    else:
        print("simple")
        response = conversation.invoke({"input":user_input})
        response = response["response"]

    print(response)



