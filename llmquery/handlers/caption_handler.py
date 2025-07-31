import sqlite3
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from llmquery.nl_processing import LLM

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

class CaptionedFiles:
    __captioned_files = None

    @staticmethod
    def get_captioned_files():
        if CaptionedFiles.__captioned_files == None:
            CaptionedFiles.__captioned_files = CaptionedFiles.get_files_from_metadata()
        return CaptionedFiles.__captioned_files

    @staticmethod
    def get_files_from_metadata():
        conn = sqlite3.connect("data/metadata.sqlite")
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, caption FROM metadata")
        rows = cursor.fetchall()
        conn.close()

        curstr = None
        captioned_files = []
        for fp, cap in rows:
            capped_file = f"{os.path.basename(fp)}: {cap}"
            if capped_file != curstr:
                curstr = capped_file
                captioned_files.append(curstr)

        return "\n".join(cf for cf in captioned_files)

def handle_caption_query(query: HumanMessage):
    captioned_files = CaptionedFiles.get_captioned_files()
    llm = LLM.get_llm()

    chain = CAPTION_PROMPT | llm
    result = chain.invoke({"captioned_files": captioned_files, "query": query})

    return result