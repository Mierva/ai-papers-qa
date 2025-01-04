import streamlit as st

# Show title and description.
st.title("📄 Document question answering")
st.write(
    "Upload a document below and ask a question about it – AI will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get only from me. "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("API Key", type="password")
if not openai_api_key:
    st.info("Please add your API key to continue.", icon="🗝️")
else:
    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()

        # Generate an answer using the OpenAI API.

        # Stream the response to the app using `st.write_stream`.
        st.write("<PH> Summary <PH>")
