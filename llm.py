import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate, 
                                    MessagesPlaceholder, 
                                    FewShotPromptTemplate, 
                                    PromptTemplate)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import answer_examples

store = {}

def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model, streaming=True)


def load_vectorstore():
    load_dotenv()
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') 
    Pinecone(api_key=PINECONE_API_KEY)

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'uaeai'

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    return database


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def build_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        '''
        [identity]
        - 당신은 국문학과 교수입니다.
        - 주어진 대화 이력과 사용자의 최근 질문을 참고하여, 이전 대화의 맥락을 몰라도 이해할 수 있도록 질문을 재구성하세요.
        - 질문에 직접 답변하지 마세요. 
        - 필요한 경우에만 질문을 다듬고, 다듬을 필요가 없으면 원래 질문을 그대로 반환하세요.
        '''
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def build_few_shot_examples() -> str:
    example_prompt = PromptTemplate.from_template("질문: {input}\n\답변: {answer}")
    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples,
        example_prompt=example_prompt,
        prefix="다음 질문에 답변하세요 :",
        suffix="질문: {input}",
        input_variables=["input"],
    )
    foramtted_few_shot_prompt = few_shot_prompt.format(input='{input}')

    return foramtted_few_shot_prompt

def get_rag_prompt():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    prompt = hub.pull('bravery/rag-prompt-01', api_key=LANGCHAIN_API_KEY)
    return prompt

def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    database = load_vectorstore()
    prompt = hub.pull('bravery/rag-prompt-01', api_key=LANGCHAIN_API_KEY)
    llm = load_llm()

    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
    qa_chain = (
        {
            'context': database.as_retriever() | format_docs,
            'input': RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain


def build_conversational_chain():
    llm = load_llm()
    database = load_vectorstore()
    retriever = database.as_retriever(search_kwargs={'k': 2})
    history_aware_retriever= build_history_aware_retriever(llm, retriever)
    
    prompt = get_rag_prompt()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain

conversational_chain = build_conversational_chain()

def stream_ai_message(user_message, session_id=None):
    ai_message = conversational_chain.stream(
        {'input': user_message},
        config={'configurable': {'session_id': session_id}},
        )
    return ai_message