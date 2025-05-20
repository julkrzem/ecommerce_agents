from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser



class RagAgent():
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

        db_filter_prompt = PromptTemplate.from_template(
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
            Each field must have an explicit operator like $eq, $in, etc. 
            If there are multiple fieds they have to be arranged with $and $or operator depending on the context.

            Context: {question}

            Neever change names of key and value only strip any additional data and organise as a valid query:
            {output}

            Return only the query filter and no furrther explaination. ONLY JSON WITH QUERY FILTER in MONGO-DB FORMAT. Remember that multiple filters have to be used in [] with $and or $or
            """
            
        )

        self.db_filter_chain = db_filter_prompt | self.llm 

    def run(self, question):

        retrived_content = ""

        needs_filtering = self.db_filter_chain.invoke({"question" : question}).content

        if "no_filter_needed" in needs_filtering:
            results = self.retriever.invoke(question)
        else:
            json_parser = SimpleJsonOutputParser()
            select_filters_chain = self.select_filter_prompt | self.llm | json_parser
            filter_db = select_filters_chain.invoke({"question" : question, "output":needs_filtering})
            print(filter_db)
            results = self.retriever.invoke(question, filter=filter_db)
        
        for res in results:
            retrived_content += f" - {res.page_content} \n"
        
        print(retrived_content)
    
        return retrived_content

# agent = RagAgent()
# answer = agent.run("What they say about Dresses in the General division?")
# print(answer)