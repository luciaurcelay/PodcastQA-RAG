import torch
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.weaviate import Weaviate
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  BitsAndBytesConfig,
  pipeline
)


def load_phi2():
    torch.set_default_device("cuda")
    # Import model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    # Define text generation pipeline
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1000,
    )
    phi2_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return phi2_llm


def load_Mistral7B_quant():
    # Define model and tokenizer
    model_name='mistralai/Mistral-7B-Instruct-v0.2'

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    # Set up quantization config
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Load pre-trained config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )

    # Define text generation pipeline (from transformers library)
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1000,
    )

    mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return mistral_llm


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


def create_chain(client):
    # Initialize retriever
    vectorstore = Weaviate(client, "Chatbot", "content")
    retriever = vectorstore.as_retriever()
    # Generate prompt template
    prompt_template = create_prompt_template()
    # Load LLM model
    llm = load_phi2()
    # Create llm chain
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain
    )

    return rag_chain