from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import CTransformers
import os

def setup_conversational_chain(vector_store):
    """
    Set up the conversational chain with local LLM
    """
    # Model configuration - using a smaller model for better compatibility
    model_config = {
        "model": "TheBloke/Llama-2-7B-Chat-GGML",  # Using GGML format for wider compatibility
        "model_file": "llama-2-7b-chat.ggmlv3.q4_0.bin",
        "model_type": "llama",
        "max_new_tokens": 512, 
        "temperature": 0.7,
        "context_length": 2048,  # Reduced for stability
        "gpu_layers": 0  # CPU only
    }
    
    # Initialize local LLM
    try:
        llm = CTransformers(
            model=model_config["model"],
            model_file=model_config["model_file"],
            model_type=model_config["model_type"],
            config={
                'max_new_tokens': model_config["max_new_tokens"],
                'temperature': model_config["temperature"],
                'context_length': model_config["context_length"],
                'gpu_layers': model_config["gpu_layers"]
            }
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to a smaller model if the main one fails
        llm = CTransformers(
            model="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGML",
            model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.ggml",
            model_type="llama",
            config={'max_new_tokens': 256, 'temperature': 0.7}
        )
    
    # Set up memory for conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )
    
    return chain