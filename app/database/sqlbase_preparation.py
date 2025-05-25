import duckdb
import pandas as pd


selected_columns = ['comment_id','clothing_id','age','rating',
                    'recommended_ind','positive_feedback_count',
                    'division_name','department_name','class_name']

df = pd.read_csv("app/data/sample.csv")
df = df[selected_columns]


for col in df.columns:
    df = df.rename(columns={col:col.replace(" ","_").lower()})


query = """DESCRIBE reviews"""

with duckdb.connect("app/database/reviews.duckdb") as con:
    con.register("temp_df", df)
    con.execute("CREATE TABLE reviews AS SELECT * FROM temp_df")
    result = con.execute(query).fetchdf()
    print(result)

# with duckdb.connect("app/database/reviews.duckdb") as con:
#     result = con.execute(query).fetchdf()
#     print(result)
