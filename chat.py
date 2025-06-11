import streamlit as st
from llm import get_ai_message

st.set_page_config(
    page_title='UAE AI 문의 챗봇',
    page_icon='🏜',
    )

st.title('🏜UAE AI 문의 챗봇🏜')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('UAE AI 관련 질문을 해주세요'):
    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.message_list.append({'role': 'user', 'content': prompt})

    with st.chat_message('ai'):
        with st.spinner('답변을 생성하는 중입니다'):
            ai_message = get_ai_message(prompt)
            st.write(ai_message)
    st.session_state.message_list.append({'role': 'ai', 'content': ai_message})