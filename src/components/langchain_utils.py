from langchain.vectorstores.weaviate import Weaviate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from . import llms 
from os.path import join, dirname
from operator import itemgetter
from dotenv import load_dotenv

dotenv_path = dotenv_path = join(dirname(dirname(dirname(__file__))), '.env')
load_dotenv(dotenv_path)

def create_chain(client, show_name, episode_name, k_top_chunks, llm_type, llm_model, memory_window):
    # Initialize retriever
    vectorstore = Weaviate(
        client, 
        "Chatbot", 
        "content", 
        attributes=["show_name", "episode_name", "row"]
        )
    # Define filters
    where_filter = get_filters(show_name, episode_name)
    # Generate prompt template
    prompt_template = create_prompt_template(llm_model)
    # Load LLM
    if llm_type == "local":
        llm = llms.load_model(llm_model)
    elif llm_type == "api":
        llm = HuggingFaceHub(
            repo_id=llm_model, 
            model_kwargs={"temperature":0.5, "max_length":512}
            )
    # Instantiate ConversationBufferMemory
    memory = ConversationBufferWindowMemory(
        k=memory_window, memory_key="chat_history", input_key="question", output_key='answer', return_messages=True)
    # Define RAG chain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        memory = memory,
        retriever = vectorstore.as_retriever(k=k_top_chunks, search_kwargs={"where_filter": where_filter}),
        verbose = True,
        combine_docs_chain_kwargs={'prompt': prompt_template},
        get_chat_history = lambda h : h,
        return_source_documents=True
    )
    return rag_chain


def get_filters(show_name, episode_name):
    filters = []
    # Add filter for show name
    if show_name != "Any":
        show_name_filter = {
            "operator": "Equal",
            "path": ["show_name"],
            "valueString": show_name,
            }
        filters.append(show_name_filter)
        # Add filter for episode name
        if episode_name != "Any":
            episode_name_filter = {
                "operator": "Equal",
                "path": ["episode_name"],
                "valueString": episode_name,
                }
            filters.append(episode_name_filter)
    return filters

def create_prompt_template(llm_model):
    if llm_model == "Open-Orca/oo-phi-1_5":
        prompt_template = phi_prompt_template()
    else:
        prompt_template = basic_prompt_template()  
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template,
    )
    return prompt_template


def basic_prompt_template():
    prompt_template = """
    ### Instruction: Answer the question delimited by triple backticks. \
    Base your answer only on the context and on the chat history, both delimited by triple backticks. \

    ### QUESTION:
    ```{question} ```


    ### CONTEXT:
    ```{context} ```

    ### CHAT_HISTORY
    ```{chat_history}```

    """
    return prompt_template


def phi_prompt_template():
    sys_prompt = """
    I am OrcaPhi. The following is my internal dialogue as an AI assistant. \
    I have no access to outside tools, news, or current events. \
    I carefully provide accurate, factual, thoughtful, nuanced answers and am brilliant at reasoning. \
    I think through my answers step-by-step to be sure I always get the right answer. \
    I think more clearly if I write out my thought process in a scratchpad manner first. \
    Take a deep breath and think calmly about everything presented.
    """
    prompt = """
    ### Instruction: Answer the following question delimited by triple backticks. \
    ### QUESTION: \
    ```{question} ``` \
    Base your answer only on the context and on the chat history, both delimited by triple backticks. \
    ### CONTEXT: \
    ```{context} ``` \
    ### CHAT_HISTORY \
    ```{chat_history}``` \
    Answer directly to the question, do NOT answer to any other question and be concise in your answer. \
    """
    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + sys_prompt + suffix
    user_format = prefix + "user\n" + prompt + suffix
    assistant_format = prefix + "assistant\n"
    prompt_template = sys_format + user_format + assistant_format
    return prompt_template
