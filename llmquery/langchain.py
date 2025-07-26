from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import ast

def get_llm():
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model='mistral',
        temperature=0
    )

    return llm

def generalize_cluster(captions: list[str], model: ChatOllama):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an imaginative assistant who concisely-summarizes descriptions of audio. 
                You are receiving a collection of descriptions.
                Each of these descriptions corresponds to different audio,
                but all of the audio is related.
                Without rehashing the descriptions, create a one-sentence summary of the similarities
                in sound evoked by the entire collection.
                Create a one- or two-word title based on your summary.
                Format your response as a Python dictionary, composed of two keys--'title' and 'summary'--
                with values corresponding respectively to the title and summary that you have generated.
                """
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | model
    result = chain.invoke(
        {
            "input": "\n".join(captions),
        }
    )
    
    return ast.literal_eval(result.content)