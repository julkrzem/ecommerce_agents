The project created in this repository is an example of a Agentic system written using LangChain, with the aim to provide insights into e-commerce data, in this case product reviews.

The repository is organized into 3 main directories:
- notebooks: (EDA) and data transformations
- app: core application logic and functionality
- tests: testing scripts

Currently, the app contains components such as chat and agents, vector database, an SQL database, and scripts for setting up and populating these databases.

Key features of the application:
- Chat that allows user interaction
- Context-aware question answering based on the data
- Agents that can query both SQL database and perform RAG on the vector database
- Main agent that delegates tasks to sub-agents and collects their responses
- "Answer agent" that combines the results and provides a final answer


This project was created as a part of my programing portfolio, based on a Kaggle dataset containing E-Commerce Clothing Reviews (https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)