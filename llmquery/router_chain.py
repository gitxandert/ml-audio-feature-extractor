from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.router import RouterRunnable

from langchain_ollama import ChatOllama

from handlers.sql_handler import handle_sql_query
from handlers.caption_handler import handle_caption_query
from handlers.embedding_handler import handle_embedding_query

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

def route_query(query: str, llm: ChatOllama, embedding: np.ndarray = None):
    routes = {
        "sql": RunnableLambda(handle_sql_query),
        "caption": RunnableLambda(handle_caption_query),
        "embedding": RunnableLambda(partial(handle_embedding_query, embedding=embedding)),
    }

    router = RouterRunnable(runnables=routes)

    router_chain = ROUTER_PROMPT | llm | (lambda x: x.strip().lower())

    query_router = router_chain | router

    result = query_router.invoke({"input": query})

    return result