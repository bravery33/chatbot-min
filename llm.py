import os

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone


load_dotenv()

def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') 

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'uaeai'

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    return database


def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    database = get_database()
    prompt = hub.pull('bravery/rag-prompt-01', api_key=LANGCHAIN_API_KEY)
    llm = get_llm()

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            'context': database.as_retriever() | format_docs,
            'question': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


def get_ai_message(user_message):
    qa_chain = get_retrievalQA()
    ai_message = qa_chain.invoke(user_message)
    return ai_message