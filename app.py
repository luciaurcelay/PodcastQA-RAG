import streamlit as st
import pandas as pd
from src.components.vector_db import create_vector_db, connect_to_vector_db, preprocess_data
from src.components.langchain_utils import create_chain
import box
import yaml
from dataclasses import dataclass



def get_show_names():
    df = pd.read_csv('src\data\podcasts.csv')
    show_names = df['show_name'].unique()
    return show_names, df

def print_sources(context, assistant_message):
    for d in context:  
        assistant_message.write("\n")
        assistant_message.write("Here are the sources of the answer:")   
        assistant_message.write(f"Show name: {d.metadata['show_name']}")
        assistant_message.write(f"Episode name: {d.metadata['episode_name']}")
        assistant_message.write(f"Minute: {d.metadata['row']}")
        assistant_message.write(f"Content: {d.page_content}")

def handle_userinput(query):
    with st.spinner("Generating answer"):
        # Run chain
        response = st.session_state.rag_chain.invoke(query)
    st.session_state.chat_history = response['chat_history']
    st.session_state.messages.append(Message(actor="user", payload=query))
    st.session_state.messages.append(Message(actor="assistant", payload=response["answer"]))
    st.chat_message("user").write(query)
    st.chat_message("assistant").write(response["answer"])
    
@dataclass
class Message:
    actor: str
    payload: str

def main():

    # Import config vars
    with open('config.yaml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    st.set_page_config(
        page_title="Podcast Chatbot",
        page_icon="🎙️"
    )
    st.header("Podcast Chatbot 🎙️")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    

    with st.sidebar:
        # Generate dropdown menus
        st.header("Time to select!")
        show_names, df = get_show_names()
        selected_show = st.selectbox('Select show name:',  ['Any'] + list(show_names))
        filtered_df = df[df['show_name'] == selected_show]
        episode_names = filtered_df['episode_name'].unique()
        selected_episode = st.selectbox('Select episode name:',  ['Any'] + list(episode_names))
        if st.button("Start querying"):
            with st.spinner("Processing"):
                # Create/initialize client
                client = connect_to_vector_db()
                # Initialize chain
                st.session_state.rag_chain = create_chain(
                    client, 
                    selected_show, 
                    selected_episode, 
                    cfg.search.search_strategy, 
                    cfg.search.alpha, 
                    cfg.k_top_chunks, 
                    cfg.llm.type, 
                    cfg.llm.model, 
                    cfg.memory.window
                    )
                
    for msg in st.session_state.messages:
        st.chat_message(msg.actor).write(msg.payload)
                
    query = st.text_input("Ask a question: ", key="user_input")

    if query:
        handle_userinput(query)


if __name__ == '__main__':
    main()