import os
import sys
import requests
import fitz  # PyMuPDF
import streamlit as st

# Add the project root (ai-papers-qa) to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Constants
API_URL = "http://127.0.0.1:8000"

# Functions
def validate_api_key(input_key, valid_key):
    """Validates the provided API key."""
    return input_key == valid_key


def extract_text_from_pdf(file):
    """Extracts text from the uploaded PDF file."""
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def get_api_response(api_url, question):
    """Sends the question to the API and retrieves the response."""
    payload = {"prompt": question, "max_length": 100}
    try:
        response = requests.post(api_url+"/chat", json=payload)
        response.raise_for_status()
        return response.json().get("response", "No response available.")
    except requests.RequestException as e:
        return f"Error communicating with API: {e}"


# Initialize Streamlit Session State for Conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = ""  # Stores the conversation history

# Streamlit Application
st.title("📄 Document Question Answering")
st.write(
    "Upload a document below and ask a question about it - AI will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get only from me."
)

# Input: API Key
api_key = st.text_input("API Key", type="password")

if not api_key:
    st.info("Please provide your API key to continue.", icon="🗝️")
    st.stop()

# Validate API Key
valid_key = st.secrets["api_key"]
if not validate_api_key(api_key, valid_key):
    st.error("Invalid API key. Please try again.", icon="⚠️")
    st.stop()

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.info("Extracting text from the PDF...")
    pdf_text = extract_text_from_pdf(uploaded_file)

    if not pdf_text.strip():
        st.warning("No text found in the PDF. Please upload a valid document.", icon="❌")
        st.stop()

    st.success("Text extraction complete!")
    with st.expander("📜 View Extracted Text"):
        st.text_area("Extracted Text", pdf_text, height=300)

    # Input: Question
    question = st.text_area(
        "Ask a question about the document:",
        placeholder="E.g., Can you summarize this document?",
    )

    if question.strip():
        # Add user's question to conversation history
        st.session_state.conversation += f"Human: {question}\n"

        # Fetch the answer
        st.info("Fetching the answer...")
        answer = get_api_response(API_URL, question)

        if "Error" in answer:
            st.error(answer, icon="❌")
        else:
            # Add the AI's response to conversation history
            st.session_state.conversation += f"AI: {answer}\n"

        # Display the updated conversation history
        st.text_area("Conversation", st.session_state.conversation, height=300)
