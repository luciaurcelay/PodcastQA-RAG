from langchain.vectorstores.weaviate import Weaviate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from llms import load_model

def create_chain(client):
    # Initialize retriever
    vectorstore = Weaviate(client, "Chatbot", "content")
    retriever = vectorstore.as_retriever()
    # Generate prompt template
    prompt_template = create_prompt_template()
    # Load LLM model
    llm = load_model("microsoft/phi-1_5")
    # Create llm chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    return rag_chain


def create_prompt_template():
    # Create prompt template
    prompt_template = """
    ### [INST] Instruction: Answer the question delimited by triple backticks. \
    Base your answer only on the context delimited by triple backticks. \

    ### QUESTION:
    ```{question} ```


    ### CONTEXT:
    ```{context} ```

    [/INST]"""

    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    return prompt_template
