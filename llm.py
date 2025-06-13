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

load_dotenv()
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def load_llm(model='gpt-4o'):
    return ChatOpenAI(model=model, streaming=True)


def load_vectorstore():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY') 
    Pinecone(api_key=PINECONE_API_KEY)

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'uaeai'

    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    return database


def build_history_aware_retriever(llm, retriever):
    contextualize_q_system_prompt = (
        '''
        [identity]
        - 당신은 사용자의 질문을 명확하게 다듬어주는 질문 재구성 전문 AI입니다.
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
    examples_str = ""
    for example in answer_examples:
        examples_str += f"질문: {example['input']}\n답변: {example['answer']}\n\n"
    return examples_str

def get_rag_prompt() -> ChatPromptTemplate:
    """
    RAG 체인을 위한 ChatPromptTemplate을 생성합니다.
    이 프롬프트는 시스템 역할, Few-shot 예제, 대화 기록,
    그리고 사용자 질문과 컨텍스트를 포함합니다.
    """
    # Few-shot 예제 문자열을 생성합니다.
    few_shot_examples_str = build_few_shot_examples()

    # LLM에게 전달할 시스템 메시지를 정의합니다.
    system_prompt_text = f"""
[identity]
- 당신은 UAE(아랍에미리트)의 AI 산업 전문가입니다.
- 주어진 문서(CONTEXT)를 바탕으로 사용자의 질문에 대해 상세하고 정확하게 한국어로 답변해야 합니다.
- 답변은 명확하고 구조화된 형식(예: 리스트, 단락)을 사용해 가독성을 높여주세요.
- 문서에 없는 내용은 답변에 포함하지 마세요.

[few_shot_examples]
다음은 좋은 질문과 답변의 예시입니다. 이 스타일을 참고하여 답변해주세요.
---
{few_shot_examples_str}---
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder("chat_history"),
        ("human", "CONTEXT:\n{context}\n\n질문:\n{input}"),
    ])

    return prompt


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