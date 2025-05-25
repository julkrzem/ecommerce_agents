from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import re
from pathlib import Path
import duckdb

class StatisticianAgent:
    def __init__(self):
        self.llm = ChatOllama(model="mistral:7b", 
                 temperature=0)
        sql_system_message = """
        Given an input question, create a syntactically correct DuckDB query to run to help find the user answer. You can order the results by a relevant column to return the most interesting examples in the database.

        Then create a syntactically correct single DuckDB query to obtain planned analysis from SQL table.
        You can order the results by a relevant column to return the most interesting examples in the database.

        Never query for all the columns from a specific table, only ask for a the few relevant columns given the analysis.

        Pay attention to use only the column names that you can see in the list_columns. Be careful to not query for columns that do not exist. Also, only use the following tables: "reviews"

        Columns:
        'comment_id': review id
        'clothing_id': id of item that is reviewed; int [0,1205]
        'age': age of the review author; int [18,99]
        'rating': rating the reviewer gave to the product; int [1,5]
        'positive_feedback_count': documenting the number of other customers who found this review positive; int
        'division_name': high level store division; str ['general', 'general petite', 'initmates']
        'department_name': product department name; str ['tops', 'dresses', 'bottoms', 'intimate', 'jackets', 'trend']
        'class_name': product type; str ['intimates', 'dresses', 'pants', 'blouses', 'knits', 'outerwear', 'lounge', 'sweaters', 'skirts', 'fine gauge', 'sleep', 'jackets', 'swim', 'trend', 'jeans', 'legwear', 'shorts', 'layering', 'casual bottoms', 'chemises']

        
        Answer with only one query.
        Remember that if there is aggregation funciton there always have to be GROUP BY of displayed columns!

        Here is the hint for the query preparation:
        {llm_message}
        """

        stat_system_message = """Given an input question, rewrite into a clear instruction for meaningful statistical analysis that will return valuable insights into topic of interest of the user. Make sure the analysis is focused on returning statistical data not only limited answers. 

        In plan include only steps that are possible to execute with SQL, but do not output any SQL, only clear instructions for analysis. Do not over complicate, and try to reduce the necessary amount of queries to minimum while maintaining the aim of user question. Do not suggest using any other programs or programing languages other than SQL.

        There is only one table: "reviews"
        Columns:
        'comment_id': review id
        'clothing_id': id of item that is reviewed; int [0,1205]
        'age': age of the review author; int [18,99]
        'rating': rating the reviewer gave to the product; int [1,5]
        'positive_feedback_count': documenting the number of other customers who found this review positive; int
        'division_name': high level store division; str ['general', 'general petite', 'initmates']
        'department_name': product department name; str ['tops', 'dresses', 'bottoms', 'intimate', 'jackets', 'trend']
        'class_name': product type; str ['intimates', 'dresses', 'pants', 'blouses', 'knits', 'outerwear', 'lounge', 'sweaters', 'skirts', 'fine gauge', 'sleep', 'jackets', 'swim', 'trend', 'jeans', 'legwear', 'shorts', 'layering', 'casual bottoms', 'chemises']

        Promote usage of statistical functions like: VAR_POP(), VAR_SAMP(), STDDEV_POP(), STDDEV_SAMP(), AVG()

        Always output with the descriptive data for example if you identified a category of the lowest value always output both name of category and the value. Avoid limiting to one result, always show few other examples to compare the statistics.
        
        Use only DuckDB functions
        The analysis you prepare should be executable with only one query!
        """

        self.safety_prompt = PromptTemplate.from_template("""You are a security guide. Make sure provided SQL is:
        - Correct to execute with DuckDB.
        - Safe to execute!
        - Only enables to query database not modify it or create any new views, tables etc.
        - There isn't a recursive query

        If the provided SQL satisfies all of the above conditions return YES. Otherwise return NO.

        Generated query: {query}
         
        Answer with YES or NO without further explanation.
        """
        )

        user_message = "Question: {question}"
        
        sql_prompt_template = ChatPromptTemplate(
            [("system", sql_system_message), ("user", user_message)]
        )

        stat_analysis_prompt = ChatPromptTemplate(
            [("system", stat_system_message), ("user", user_message)]
        )

    
        self.sql_query_chain = sql_prompt_template | self.llm

        self.stat_analysis_chain = stat_analysis_prompt | self.llm

        self.safety_check = self.safety_prompt | self.llm
    
    def prepare_stat_analysis(self, question: str) -> str:
        result = self.stat_analysis_chain.invoke({"question":question})
        # print(result)
        return result.content

    def prepare_sql_query(self, question: str, llm_instruction: str) -> str:
        result = self.sql_query_chain.invoke({"question": question, "llm_message":llm_instruction})
        # print(result)
        return result.content
    
    def check_query_regex(self, query: str)->int:
        words = ["DELETE", "INSERT", "UPDATE", "CREATE", "RECURSIVE", "WITH"]
        unsafe_match = re.findall("|".join(words),query)
        return len(unsafe_match)
    
    def check_query_llm(self, query: str)->int:
        result = self.safety_check.invoke({"query": query}).content
        return result
    
    def execute_query(self, query: str)->str:
        database_path = Path(__file__).resolve().parent.parent / "database" / "reviews.duckdb"
        with duckdb.connect(str(database_path)) as con:
            result = con.execute(query).fetchdf()
        answer = result.to_string()
        return answer

    def run(self, question: str)->str:
        llm_answ = self.prepare_stat_analysis(question)
        llm_answ_2 = self.prepare_sql_query(question, llm_answ)

       
        if len(re.findall(r'```sql(.*?)```', llm_answ_2, re.DOTALL)) > 0:
            extracted_sql = re.findall(r'```sql(.*?)```', llm_answ_2, re.DOTALL)[0].strip()
        else:
            extracted_sql = re.findall(r'SELECT(.*?)\;', llm_answ_2, re.DOTALL)[0]
            extracted_sql = "SELECT"+extracted_sql

        print(extracted_sql)

        if self.check_query_regex(extracted_sql)==0:

            safety_result = self.check_query_llm(extracted_sql)
            if "YES" in safety_result:
                try:
                    return self.execute_query(extracted_sql)
                except:
                    return ""
        else:
            return "Table modifications are not allowed or query is invalid"