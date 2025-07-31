from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.router import RouterRunnable
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from llmquery.nl_processing import LLM

from llmquery.handlers.sql_handler import handle_sql_query
from llmquery.handlers.caption_handler import handle_caption_query
from llmquery.handlers.embedding_handler import handle_embedding_query

from functools import partial

import numpy as np

ROUTER_PROMPT = PromptTemplate.from_template("""
You are a router that decides how to process user questions about audio files.
                                             
Classify the question into one of:
- sql: if the query is about structured metadata like duration, cluster, domain, etc. (e.g."Which files are the longest?"; "How many clusters are there?")
- caption: if the query asks what is happening or present in the audio (e.g. "Which files feature people singing?"; "Which files are field recordings?")
- embedding: if the query asks to compare provided audio (in the form of an array) to stored audio (e.g. "Which files sound like this one?", accompanied by an array)
                                             
Return **only one** of these three words: sql, caption, embedding

Question: {input}
Route:
""")

def route_query(query: str, embedding: np.ndarray = None):
    routes = {
        "sql": RunnableLambda(handle_sql_query),
        "caption": RunnableLambda(handle_caption_query),
        "embedding": RunnableLambda(partial(handle_embedding_query, embedding=embedding)),
    }

    router = RouterRunnable(runnables=routes)
    llm = LLM.get_llm()

    router_chain = ROUTER_PROMPT | llm

    message = [HumanMessage(query)]

    route_key = router_chain.invoke({"input": message})

    formatted_key = route_key.content.replace(" ", "")
    print(f"Route key = {formatted_key}")
    
    query_router = router.invoke({"key": formatted_key, "input": message})

    return query_router.content