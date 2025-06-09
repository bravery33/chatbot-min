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
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'uaeai'

# loader = PyPDFLoader('25중동AI붐중심UAEAI시장동향과기업진출전략.pdf')

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1500,
#     chunk_overlap=200,
# )

# document_list = loader.load_and_split(text_splitter=text_splitter)

# with open("all_pages.txt", "w", encoding="utf-8") as f:
#     for doc in document_list:
#         f.write(f"--- 페이지 {doc.metadata['page_label']} ---\n")
#         f.write(doc.page_content + "\n\n")

# database =  PineconeVectorStore.from_documents(
#     documents=document_list,
#     embedding=embedding,
#     index_name=index_name,
# )

database = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding,
)

llm = ChatOpenAI(model='gpt-4o')
prompt = hub.pull('rlm/rag-prompt')

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

print(qa_chain.invoke("UAE의 AI관련 투자금액은?"))
