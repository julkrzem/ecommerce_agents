from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser


class RagAgent:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:33m")
        self.vector_store = Chroma(
            collection_name="ecommerce_reviews",
            embedding_function=self.embeddings,
            persist_directory="./app/vectorstore/chroma_db",
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        self.llm = ChatOllama(model="mistral:7b", 
                 temperature=0)

        self.db_filter_prompt = PromptTemplate.from_template(
            """You are a smart assistant. Your job is to decide if the database should be filtered to answer the user question.

            Database fields to filter: 
            'clothing_id': ID of item that is reviewed; int [0,1205]
            'age': age of the review author; int [18,99]
            'rating': rating the reviewer gave to the product; int [1,5]
            'division_name': store division; str [General, General Petite, Initmates]
            'department_name': store department; str [Tops, Dresses, Bottoms, Intimate, Jackets, Trend]
            'class_name': product type; str [Intimates, Dresses, Pants, Blouses, Knits, Outerwear,Lounge, Sweaters, Skirts, Fine gauge, Sleep, Jackets,Swim, Trend, Jeans, Legwear, Shorts, Layering,Casual bottoms, Chemises]

            Question: {question}

            If the question requires filtering answer only with: filter key and value:
            for example if requires filtering by product type: "class_name": "Pants"
            If it doesn't require filtering answer "no_filter_needed".
            
            Do not provide any further explanation only the filter key value or no_filter_needed.
            Your answer need to be unambiguous decision."""
            
        )
        
        self.select_filter_prompt = PromptTemplate.from_template(
            """Your job is to prepare a MongoDB syntax for filtering.
            If there are multiple fieds they have to be arranged with and/or operator depending on the context.
            If there is only one field to filter by never use and/or

            Context: {question}

            Neever change names of key and value only strip any additional data and organise as a valid query:
            {output}

            Return only the query filter and no furrther explaination.

            {output_format}

            Answer with JSON WITH ONLY QUERY FILTER in MongoDB Query Language (MQL).
            """
            
        )

    def needs_filtering(self, question):
        db_filter_chain = self.db_filter_prompt | self.llm 
        return db_filter_chain.invoke({"question" : question}).content
    
    def run(self, question):

        retrived_content = ""

        filtering = self.needs_filtering(question)

        if "no_filter_needed" in filtering:
            results = self.retriever.invoke(question)
        else:
            json_parser = JsonOutputParser()

            output_format = '''
            Output format to filter by multiple fields:
            {"$and": [
                {"field_1": {"$eq": "Value_1"}},
                {"field_2": {"$eq": "Value_2"}}
                ]
            }

            Output format if filter by one field:
            {"field_1": {"$eq": "Value_1"}}
            '''

            select_filters_chain = self.select_filter_prompt | self.llm | json_parser
            filter_db = select_filters_chain.invoke({"question" : question, "output":filtering, "output_format": output_format})
            print(filter_db)
            results = self.retriever.invoke(question, filter=filter_db)
        
        for res in results:
            retrived_content += f" - {res.page_content} \n"
        
        print(retrived_content)
    
        return retrived_content
    
# agent = RagAgent()
# print(agent.run("What they say about Dresses?"))
# print(agent.run("What they say about Knits in the General Petite division?"))