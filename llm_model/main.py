from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model Configuration
import os
os.environ['HF_HOME'] = 'D:/Projects/MLOPS_proj/llm/cache'  # Cache directory

# Load Sentence Transformer for Embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLaMA Model and Tokenizer
model_name = "Qwen/QwQ-32B-Preview"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define Input Schema
class Query(BaseModel):
    question: str
    document: str  # This will be raw text from the PDF

# Utility Functions
def extract_text_from_pdf(pdf_file_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_relevant_context(question, document, top_k=5):
    """Find the most relevant context from the document."""
    sentences = document.split('. ')
    embeddings = embedder.encode(sentences)
    question_embedding = embedder.encode([question])

    # Compute cosine similarity
    similarities = cosine_similarity(question_embedding, embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Combine top sentences as context
    context = " ".join([sentences[i] for i in top_indices])
    return context

def generate_answer_with_llama(question, context):
    """Generate an answer using the LLaMA model."""
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # Generate answer
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return answer

# API Endpoints
@app.post("/answer")
async def answer_question(query: Query):
    """Handle question answering requests."""
    # Extract relevant context
    context = get_relevant_context(query.question, query.document)

    # Generate answer using LLaMA
    answer = generate_answer_with_llama(query.question, context)
    return {"answer": answer}
