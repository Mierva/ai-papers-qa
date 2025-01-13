import sys
import os

# Add the project root (ai-papers-qa) to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import requests
import fitz  # PyMuPDF

def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

API_URL = "http://127.0.0.1:8000"

st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it - AI will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get only from me. "
)

token = st.text_input("API Key", type="password")
if not token:
    st.info("Please add your API key to continue.", icon="🗝️")
else:
    isValid = True if token==st.secrets["api_key"] else False

    if not isValid:
        st.write('Wrong token buddy')
    else:
        uploaded_file = st.file_uploader(
            "Upload a PDF", type=("pdf")
        )

        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Can you give me a short summary?",
            disabled=not uploaded_file,
        )

        if uploaded_file and question:
            print(f'question: {question}')

            # Extract text
            st.info("Extracting text from PDF...")
            pdf_text = extract_text_from_pdf(uploaded_file)

            if not pdf_text.strip():
                st.warning("No text found in the PDF!")

            st.success("Text extraction complete!")
            st.text_area("Extracted Text", pdf_text + "...", height=300)

            data = {"prompt": str(question),
                    "max_length": 100
                    }

            output = requests.post(url=f'{API_URL}/generate', json=data)
            print(f'post_response: {output.json()}\n')
            # model_response = requests.get(url=f'{API_URL}/output/{output_id}', data={output_id})

            st.write(f"{output.json()['output']}")
