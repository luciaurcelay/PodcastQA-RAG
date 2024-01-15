import torch
from langchain_community.llms import HuggingFacePipeline
from transformers import (
  AutoTokenizer,
  AutoModelForCausalLM,
  BitsAndBytesConfig,
  pipeline
)
from langchain.vectorstores.weaviate import Weaviate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough

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
    ### Instruction: Answer the question delimited by triple backticks. \
    Base your answer only on the context delimited by triple backticks. \

    ### QUESTION:
    ```{question} ```


    ### CONTEXT:
    ```{context} ```

    """

    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    return prompt_template

def load_model(model_name):
    torch.set_default_device("cuda")
    # Import model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Define text generation pipeline
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=1000,
        device=0
    )
    model_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return model_pipeline


def load_quantized_model(model_name):
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

    llm_pipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

    return llm_pipeline