import streamlit as st
import pandas as pd
from src.components.vector_db import create_vector_db, connect_to_vector_db
from src.components.langchain_utils import create_chain
import box
import yaml


def get_show_names():
    df = pd.read_csv('src\data\podcasts.csv')
    show_names = df['show_name'].unique()
    return show_names, df


def print_sources(response):
    with st.expander("See sources"):
        for d in response["source_documents"]:  
            with st.chat_message("assistant"):
                st.markdown(f"Show name: {d.metadata['show_name']}")
                st.markdown(f"Episode name: {d.metadata['episode_name']}")
                st.markdown(f"Minute: {d.metadata['row']}")
                st.markdown(f"Content: {d.page_content}")


def main():
    with open('config.yaml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    st.set_page_config(
        page_title="Podcast Chatbot",
        page_icon="🎙️"
    )
    st.header("Podcast Chatbot 🎙️")

    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    query = st.chat_input("Ask a question")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Generating answer"):
            response = st.session_state.rag_chain.invoke(query)
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            print_sources(response)
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    with st.sidebar:
        st.header("Select show and episode 🎧")
        show_names, df = get_show_names()
        selected_show = st.selectbox('Select show name:',  ['Any'] + list(show_names))
        filtered_df = df[df['show_name'] == selected_show]
        episode_names = filtered_df['episode_name'].unique()
        selected_episode = st.selectbox('Select episode name:',  ['Any'] + list(episode_names))
        if st.button("Start querying"):
            with st.spinner("Loading"):
                # Create/initialize client
                client = connect_to_vector_db()
                # client = create_vector_db(cfg.chunking.chunk_size, cfg.chunking.chunk_overlap)
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
            st.subheader("Go!")
                

if __name__ == '__main__':
    main()