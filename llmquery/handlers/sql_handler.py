from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent

from langchain_core.messages import HumanMessage

from llmquery.nl_processing import LLM

class SQLAgent:
    __executor = None

    @staticmethod
    def build_sql_agent():
        llm = LLM.get_llm()
        db = SQLDatabase.from_uri("sqlite:///data/metadata.sqlite")
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        system_message = prompt_template.format(dialect="SQLite", top_k=5)

        agent_executor = create_react_agent(llm, toolkit.get_tools(), prompt=system_message)
        
        return agent_executor

    @staticmethod
    def invoke(query: HumanMessage):
        if SQLAgent.__executor == None:
            SQLAgent.__executor = SQLAgent.build_sql_agent()
        return SQLAgent.__executor.invoke({"input": query})

def handle_sql_query(query: HumanMessage):
    result = SQLAgent.invoke(query)
    return result