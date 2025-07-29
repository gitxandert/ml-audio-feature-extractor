import sqlite3
from langchain_core.prompts import PromptTemplate

from nl_processing import LLM

CAPTION_PROMPT = PromptTemplate.from_template("""
You are helping search audio files based on their descriptions.
                                              
Here is a list of files with their descriptions:
{captioned_files}

Refer to this list to answer the following question: {query}

**Respond only with file names whose descriptions match the question. Do not explain your reasoning.**

For example, given the following list...
birdsounds.wav: "Birds are singing in a field"
pigsounds.wav: "Pigs are oinking in the mud"
fishsounds.wav: "Fish are splashing in the water"

... and the question, "Which files feature bird sounds?"

... your response should be:
birdsounds.wav                                             
""")

def handle_caption_query(query: str):
    conn = sqlite3.connect("data/metadata.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, caption FROM metadata")
    rows = cursor.fetchall()
    conn.close()

    captioned_files = "\n".join(
        f"{file_path}: {caption}"
        for file_path, caption in rows
        if caption
    )

    llm = LLM.get_llm()

    chain = CAPTION_PROMPT | llm
    result = chain.invoke({"captioned_files": captioned_files, "query": query})

    return result