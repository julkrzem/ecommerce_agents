from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages.ai import AIMessage
import json

class RagAgent:
    def __init__(self, memory):
        self.memory = memory
        self.embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:33m")
        self.vector_store = Chroma(
            collection_name="ecommerce_reviews",
            embedding_function=self.embeddings,
            persist_directory="./app/database/chroma_db",
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        self.llm = ChatOllama(model="mistral:7b", 
                 temperature=0)

        self.db_filter_prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to construct the most suitable yet simple filter to answer the user question.

            Database fields to filter: 
            'clothing_id': ID of item that is reviewed; int [0,1205]
            'age': age of the review author; int [18,99]
            'rating': rating the reviewer gave to the product; int [1,5]
            'division_name': high level store division; str [General, General Petite, Initmates]
            'department_name': product department name; str [Tops, Dresses, Bottoms, Intimate, Jackets, Trend]
            'class_name': product type; str [Intimates, Dresses, Pants, Blouses, Knits, Outerwear,Lounge, Sweaters, Skirts, Fine gauge, Sleep, Jackets,Swim, Trend, Jeans, Legwear, Shorts, Layering,Casual bottoms, Chemises]

            Question: {question}

            Do not propose filters that include entire range of values 
            Wrong answer example: rating=1,2,3,4,5 explaantion: rating is always a number in range 1-5
            
            If it doesn't require filtering answer "no_filter_needed".

            Propose a filter construction in Chroma DB Query Language.
            Filter may not be necesairy for some question, if base do not require filtering answer "no_filter_needed"
            
            Do not provide any further explanation only the filter or no_filter_needed.

            Your answer need to be unambiguous decision.
            """
            
        )


        self.correct_filter_prompt = PromptTemplate.from_template(
            """Your job is to prepare a Chroma DB syntax for filtering.
            Never change names of key and value only strip any additional data and organize as a valid query.

            {invalid_query}
            Return only the correct query filter and no further explanation.

            {output_format}

            Answer ONLY with JSON WITH QUERY FILTER in Chroma DB Query Language.
            Your answer need to be unambiguous decision.
            Chroma db expects to have exactly one operator in query!
            """
        )

    def needs_filtering(self, question):
        db_filter_chain = self.db_filter_prompt | self.llm 
        return db_filter_chain.invoke({"question" : question}).content
    
    def correct_query(self, invalid_query, output_format):
        correct_filter_chain = self.correct_filter_prompt | self.llm 
        return correct_filter_chain.invoke({"invalid_query" : invalid_query, "output_format":output_format}).content.strip()

    
    def run(self, question):

        retrived_content = ""

        filtering = self.needs_filtering(question)

        if not "no_filter_needed" in filtering:
            output_format = """
            Output message in this format if filter consists of one column:
            {"column_1": {"$eq": "Value_1"}}

            Output message in this format if filter consists of multiple columns:
            {"$and": [
                {"column_1": {"$eq": "Value_1"}},
                {"column_2": {"$eq": "Value_2"}}
                ]
            }
            
            WARNING
            Chroma db expects to have exactly one operator in query!
            """
            filter_db = self.correct_query(filtering, output_format)
            
            try: 
                parsed_json = json.loads(filter_db)
                results = self.retriever.invoke(question, filter=parsed_json)
                print("filter used")
            except:
                results = self.retriever.invoke(question)
                print("filter not used")

        else:
            results = self.retriever.invoke(question)

        for res in results:
            retrived_content += f" - {res.page_content} \n"
        
        self.memory.chat_memory.add_message(AIMessage(retrived_content))
   
        return retrived_content