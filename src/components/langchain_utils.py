from langchain.vectorstores.weaviate import Weaviate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from . import llms


def create_chain(client):
    # Initialize retriever
    vectorstore = Weaviate(client, "Chatbot", "content", attributes=["metadata"])
    retriever = vectorstore.as_retriever(k=3)
    # Generate prompt template
    prompt_template = create_prompt_template()
    # Load LLM model
    llm = llms.load_model("Open-Orca/oo-phi-1_5")
    
    # Create llm chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )
    return rag_chain


def create_prompt_template():
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
    Use only the following context delimited by triple backticks to answer the question. \
    ### CONTEXT: \
    ```{context} ``` \
    Answer directly to the question, do NOT answer to any other question and be concise in your answer. \
    """
    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + sys_prompt + suffix
    user_format = prefix + "user\n" + prompt + suffix
    assistant_format = prefix + "assistant\n"
    prompt_template = sys_format + user_format + assistant_format

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    return prompt_template


def create_basic_prompt_template():
    prompt_template = """
    ### Instruction: Answer the question delimited by triple backticks. \
    Base your answer only on the context delimited by triple backticks. \

    ### QUESTION:
    ```{question} ```


    ### CONTEXT:
    ```{context} ```

    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    return prompt_template
