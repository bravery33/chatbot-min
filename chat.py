import streamlit as st
import uuid
from llm import stream_ai_message

st.set_page_config(
    page_title='UAE의 AI 산업관련 문의 챗봇',
    page_icon='👳‍♂️',
    )

st.title('UAE의 AI산업 관련 문의 챗봇👳‍♂️')

query_params = st.query_params

if 'session_id' in query_params:
    session_id = query_params['session_id']
else:
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})

if 'session_id' not in st.session_state:
    st.session_state.session_id = session_id

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

if prompt := st.chat_input('UAE의 AI 산업과 관련된 질문을 해주세요'):
    with st.chat_message('user'):
        st.write(prompt)
    st.session_state.message_list.append({'role': 'user', 'content': prompt})

    with st.chat_message('ai'):
        with st.spinner('오아시스를 찾는 중입니다'):
            ai_message = stream_ai_message(
                user_message=prompt,
                session_id=st.session_state.session_id
            )
            ai_message = st.write_stream(ai_message)
    st.session_state.message_list.append({'role': 'ai', 'content': ai_message})