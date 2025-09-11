# PDF Chatbot with Local Models

A Streamlit application that allows you to chat with your PDF documents using locally running language models.

## Features

- Upload and process PDF documents
- Ask questions about the document content
- All processing happens locally (no external APIs)
- Chat history maintained during session
- Source document references

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv pdf_chatbot_env
   source pdf_chatbot_env/bin/activate  # On Windows: pdf_chatbot_env\Scripts\activate



# File Structure:
   pdf-chatbot/ <br>
├── app.py<br>
├── requirements.txt<br>
├── utils/<br>
│   ├── __init__.py<br>
│   ├── pdf_processor.py<br>
│   ├── embeddings.py<br>
│   └── chain_setup.py<br>
└── README.md<br>


# Install dependencies:

bash
pip install -r requirements.txt
Usage
Run the application:

bash
streamlit run app.py
Upload a PDF file using the sidebar

Click "Process PDF" to extract and index the content

Start asking questions about the document

# Note on First Run
The first time you run the application, it will download several GB of model files:

Sentence transformer model for embeddings (~80MB)

Llama 2 7B Chat model (~4GB)

This download only happens once. Subsequent runs will use the cached models.

# Model Customization
You can change the LLM model by modifying the setup_conversational_chain function in utils/chain_setup.py. Many GGUF format models are available on Hugging Face Hub.

Hardware Requirements
At least 8GB RAM recommended

The application runs on CPU but can be configured for GPU with CUDA support

text

## How to Run:

1. Create the project directory and files as shown above
2. Install the requirements: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Important Notes:

1. The first run will download models which may take time depending on your internet connection
2. The Llama-2-7B model requires about 4GB of disk space
3. For better performance, consider using a smaller model if you have limited RAM
4. You can change the model by modifying the model path in `utils/chain_setup.py`

This implementation provides a complete, self-contained PDF chatbot that runs entirely on your local machine without any external API dependencies.

