import streamlit as st
import pdfplumber
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# HTML and CSS for custom chat interface
css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #2b313e;
    color: #fff;
    width: 95%;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    padding: 0 1.5rem;
}
</style>
'''

# Bot's answer template with avatar
bot_template = '''
<div class="chat-message">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 53px; max-width: 53px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{}</div>
</div>
'''

# User's question template with avatar
user_template = '''
<div class="chat-message">
    <div class="avatar">
        <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png" style="max-height: 53px; max-width: 53px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{}</div>
</div>
'''

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.strip() + "\n"
    return text

# Function to create embeddings using SentenceTransformer model
def create_embeddings(text_chunks):
    if not text_chunks:
        logging.error("No text chunks provided for embedding creation.")
        return []

    model = SentenceTransformer("all-mpnet-base-v2")
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    logging.info(f"Created {len(embeddings)} embeddings.")
    return embeddings

# Function to perform similarity search using embeddings
def search_answer(user_question, embeddings, text_chunks):
    if len(embeddings) == 0:
        logging.error("No embeddings available for search.")
        return "No data available to answer the question."

    model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)
    most_similar_idx = similarities.argmax().item()
    return text_chunks[most_similar_idx]

# Function to split text into manageable chunks
def get_text_chunks(text, chunk_size=500, overlap=100):
    if not text:
        logging.error("No text provided to split into chunks.")
        return []

    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(' '.join(words[i:i + chunk_size]))

    logging.info(f"Split text into {len(chunks)} chunks.")
    return chunks

# Function to store chat history in a file
def store_chat_history(chat_history):
    with open("chat_history.pkl", "wb") as f:
        pickle.dump(chat_history, f)

# Function to load chat history from a file
def load_chat_history():
    if os.path.exists("chat_history.pkl"):
        with open("chat_history.pkl", "rb") as f:
            return pickle.load(f)
    return []

# Function to initialize the interface with chat history management options
def initialize_interface():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with multiple PDFs :books:")

    # Chat History Management Options
    st.sidebar.subheader("Chat History Management")
    if st.sidebar.button("Save Chat History"):
        store_chat_history(st.session_state.chat_history)
    if st.sidebar.button("Load Chat History"):
        st.session_state.chat_history = load_chat_history()
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []

    # Display uploaded PDFs in the sidebar
    pdf_files = st.sidebar.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
    if pdf_files:
        st.sidebar.subheader("Uploaded PDFs")
        for pdf in pdf_files:
            display_pdf_sidebar(pdf)

# Function to handle user inputs and search for answers
def handle_userinput(user_question, embeddings, text_chunks, chat_history):
    # Check if embeddings or text_chunks are empty or None
    if embeddings is None or text_chunks is None or len(embeddings) == 0 or len(text_chunks) == 0:
        st.error("Embeddings have not been created or text chunks are empty. Please upload PDF files and try again.")
        return

    # Use SentenceTransformer model for similarity search
    model = SentenceTransformer("all-mpnet-base-v2")
    query_embedding = model.encode(user_question, convert_to_tensor=True)
    
    # Perform similarity search using embeddings
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)
    
    # Handle case when no similar text chunk found
    if similarities.max().item() < 0.5:  # Adjust threshold as needed
        st.error("Question not relevant to the uploaded PDFs. Please ask another question.")
        return
    
    most_similar_idx = similarities.argmax().item()

    # Retrieve the most relevant text chunk based on similarity
    relevant_text = text_chunks[most_similar_idx]

    # Provide the relevant text as the answer
    answer = relevant_text[:1000]  # Limit to first 1000 characters

    # Log and store chat history
    chat_history.append((user_question, answer))
    store_chat_history(chat_history)

    # Display user question and bot's answer in the chat interface
    st.markdown(user_template.format(user_question), unsafe_allow_html=True)
    st.markdown(bot_template.format(answer), unsafe_allow_html=True)

# Function to display PDF pages in the sidebar
def display_pdf_sidebar(pdf_file):
    pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        image_bytes = page.get_pixmap().tobytes()
        st.sidebar.image(image_bytes, caption=f"Page {page_num + 1} of {pdf_file.name}", use_column_width=True)

def main():
    initialize_interface()

    # Load chat history if it exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()

    # Upload PDF files
    pdf_files = st.file_uploader("Upload your PDF files", type='pdf', accept_multiple_files=True)

    # Process uploaded PDFs and create embeddings
    embeddings = []
    text_chunks = []
    if pdf_files:
        text = extract_text_from_pdf(pdf_files)
        text_chunks = get_text_chunks(text)
        if text_chunks:
            embeddings = create_embeddings(text_chunks)
        else:
            st.error("Failed to extract text from the uploaded PDFs.")

    # User question input and handle user input
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_userinput(user_question, embeddings, text_chunks, st.session_state.chat_history)

    # Display chat history
    st.subheader("Chat History")
    st.write('<div class="chat-container">', unsafe_allow_html=True)
    for question, answer in st.session_state.chat_history:
        st.markdown(user_template.format(question), unsafe_allow_html=True)
        st.markdown(bot_template.format(answer), unsafe_allow_html=True)
    st.write('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
