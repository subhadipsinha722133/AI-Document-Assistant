import streamlit as st
import os
from utils.pdf_processor import extract_text_from_pdf, chunk_text
from utils.embeddings import create_embeddings, create_vector_store
from utils.chain_setup import setup_conversational_chain
from utils.pdf_diagnostic import analyze_pdf, extract_images_from_pdf
from langchain.docstore.document import Document
import tempfile
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot with Local Models",
    page_icon="📄",
    layout="wide"
)

# Initialize session state variables
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "ocr_used" not in st.session_state:
    st.session_state.ocr_used = False
if "pdf_analysis" not in st.session_state:
    st.session_state.pdf_analysis = None
if "sample_images" not in st.session_state:
    st.session_state.sample_images = []

# Sidebar for file upload
with st.sidebar:
    st.title("PDF Chatbot Settings")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Upload a PDF file", 
        type="pdf",
        help="Select a PDF file to process"
    )
    
    # Advanced options
    st.markdown("### Advanced Options")
    use_ocr = st.checkbox("Use OCR (for scanned documents)", value=True)
    high_dpi = st.checkbox("High resolution OCR (slower)", value=False)
    show_diagnostics = st.checkbox("Show PDF diagnostics", value=True)
    
    process_button = st.button("Process PDF", disabled=not uploaded_file)
    
    if process_button and uploaded_file:
        # Reset state
        st.session_state.processed = False
        st.session_state.pdf_analysis = None
        st.session_state.sample_images = []
        
        # Analyze PDF first
        if show_diagnostics:
            with st.spinner("Analyzing PDF structure..."):
                st.session_state.pdf_analysis = analyze_pdf(uploaded_file)
                uploaded_file.seek(0)  # Reset file pointer
        
        # Extract sample images for problematic PDFs
        if (st.session_state.pdf_analysis and 
            st.session_state.pdf_analysis["has_images"] and 
            not st.session_state.pdf_analysis["has_text"]):
            with st.spinner("Extracting sample images..."):
                st.session_state.sample_images = extract_images_from_pdf(uploaded_file, max_pages=2)
                uploaded_file.seek(0)
        
        # Process PDF
        with st.spinner("Processing PDF..."):
            try:
                # Extract text from PDF
                text = extract_text_from_pdf(uploaded_file)
                
                if not text.strip():
                    st.error("Could not extract text from PDF using any method.")
                    st.info("This PDF may be: 1) Password protected 2) Pure images without text 3) Malformed")
                    
                    # Show manual workaround
                    with st.expander("Manual Workaround Options"):
                        st.markdown("""
                        If automatic extraction fails, you can:
                        1. **Convert PDF to text**: Use Adobe Acrobat or online tools
                        2. **Copy text manually**: Select text and copy-paste
                        3. **Use alternative OCR**: Try Google Drive or other OCR services
                        4. **Check if PDF is protected**: Some PDFs restrict text extraction
                        """)
                else:
                    # Display text extraction stats
                    char_count = len(text)
                    word_count = len(text.split())
                    st.info(f"Extracted {char_count} characters ({word_count} words) from PDF")
                    
                    # Show text preview
                    with st.expander("View extracted text preview"):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)
                    
                    # Chunk the text
                    chunks = chunk_text(text)
                    st.info(f"Created {len(chunks)} text chunks")
                    
                    # Create embeddings
                    with st.spinner("Creating embeddings..."):
                        embeddings = create_embeddings()
                    
                    # Create vector store
                    with st.spinner("Building vector database..."):
                        vector_store = create_vector_store(chunks, embeddings)
                        st.session_state.vector_store = vector_store
                    
                    # Set up conversation chain
                    with st.spinner("Loading language model..."):
                        st.session_state.conversation_chain = setup_conversational_chain(vector_store)
                    
                    st.session_state.processed = True
                    st.success("PDF processed successfully! You can now ask questions.")
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This chatbot uses local models to answer questions about your PDF. "
        "It includes advanced OCR for challenging documents."
    )

# Main content area
st.title("PDF Chatbot with Local Models")
st.markdown("Ask questions about your PDF document")

# Show diagnostics if available
if st.session_state.pdf_analysis and show_diagnostics:
    with st.expander("PDF Analysis Report"):
        st.json(st.session_state.pdf_analysis)
        
        if st.session_state.pdf_analysis["is_encrypted"]:
            st.error("⚠️ PDF appears to be encrypted or password protected")
        if st.session_state.pdf_analysis["has_images"] and not st.session_state.pdf_analysis["has_text"]:
            st.warning("📷 PDF contains images but no extractable text")

# Show sample images if available
if st.session_state.sample_images:
    with st.expander("Sample Images from PDF"):
        for page_num, img_index, image in st.session_state.sample_images:
            st.image(image, caption=f"Page {page_num + 1}, Image {img_index + 1}")

if st.session_state.processed:
    if st.session_state.ocr_used:
        st.warning("⚠️ This PDF was processed using OCR. Text accuracy may be lower than native text PDFs.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDF..."):
    if not st.session_state.processed:
        st.warning("Please upload and process a PDF first.")
    else:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from the chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from the chain
                    response = st.session_state.conversation_chain({"question": prompt})
                    answer = response["answer"]
                    
                    # Display assistant response
                    st.markdown(answer)

                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Show source documents (optional)
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                            
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Manual text input fallback
if not st.session_state.processed and uploaded_file:
    with st.expander("Manual Text Input (Fallback)"):
        st.warning("Automatic extraction failed. You can manually input text from the PDF:")
        manual_text = st.text_area("Paste text from PDF here:", height=200)
        if st.button("Process Manual Text") and manual_text.strip():
            try:
                # Chunk the manual text
                chunks = chunk_text(manual_text)
                
                # Create embeddings
                embeddings = create_embeddings()
                
                # Create vector store
                vector_store = create_vector_store(chunks, embeddings)
                st.session_state.vector_store = vector_store
                
                # Set up conversation chain
                st.session_state.conversation_chain = setup_conversational_chain(vector_store)
                
                st.session_state.processed = True
                st.success("Manual text processed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing manual text: {str(e)}")

# Clear chat button
if st.session_state.chat_history:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        if st.session_state.conversation_chain:
            st.session_state.conversation_chain.memory.clear()
        st.rerun()

# Reset button
if st.sidebar.button("Reset Everything"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()