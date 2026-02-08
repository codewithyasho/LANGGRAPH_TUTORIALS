from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from main import chatbot, get_threads_in_db
import streamlit as st
import uuid


# =================== utility functions ===================
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_chat(thread_id):
    return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values["messages"]


def get_thread_preview(thread_id):
    """Generate a user-friendly preview for a chat thread"""
    try:
        messages = load_chat(thread_id)
        if messages:
            # Get the first user message
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    preview = msg.content[:40]  # First 40 characters
                    if len(msg.content) > 40:
                        preview += "..."
                    return preview
        return f"Chat {str(thread_id)[:8]}..."
    except:
        return f"Chat {str(thread_id)[:8]}..."


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_threads_in_db()

add_thread(st.session_state['thread_id'])


# =================== SIDEBAR ===================
st.sidebar.title('Langgraph Chatbot')

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Your Chats")

# show most recent chats first
for thread_id in st.session_state['chat_threads'][::-1]:
    preview = get_thread_preview(thread_id)
    if st.sidebar.button(preview, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_chat(thread_id)

        temp_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': message.content})

        st.session_state['message_history'] = temp_messages


# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# {'role': 'user', 'content': 'Hi'}
# {'role': 'assistant', 'content': 'Hello'}

user_input = st.chat_input('Type here')


if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append(
        {'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # first add the message to message_history
    with st.chat_message('assistant'):

        def ai_stream_only():
            for chunk in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # stream_mode="messages" returns (message, metadata) tuples
                if isinstance(chunk, tuple):
                    message = chunk[0]
                else:
                    message = chunk

                if isinstance(message, AIMessage) and message.content:
                    # only yield AIMessage chunks, ignore tool chunks
                    yield message.content

        ai_msg = st.write_stream(ai_stream_only())

    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_msg}
    )
