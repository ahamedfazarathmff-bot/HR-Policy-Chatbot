import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import os
import tempfile
from pypdf import PdfReader

load_dotenv()

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, index, groq_client

model, index, groq_client = load_models()

# PDF process function
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    pages = [page.extract_text() for page in reader.pages if page.extract_text()]

    # Chunk
    chunks = []
    for text in pages:
        start = 0
        while start < len(text):
            end = min(start + 900, len(text))
            chunks.append(text[start:end].strip())
            start = end - 150

    # Embed and store
    embeddings = model.encode(chunks).tolist()
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{uploaded_file.name}_{i}",
            "values": embedding,
            "metadata": {"text": chunk, "source": uploaded_file.name}
        })

    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i:i+100])

    return len(chunks)

def get_answer(question: str, chat_history: list):
    # Embed question
    question_embedding = model.encode(question).tolist()

    # Search Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n\n"

    # Build messages with chat history
    messages = [
        {
            "role": "system",
            "content": "You are an HR Policy assistant. Answer questions based only on the provided context. Be clear and concise."
        }
    ]

    # Add chat history
    for chat in chat_history[-6:]:  # Last 6 messages
        messages.append({"role": chat["role"], "content": chat["content"]})

    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    return response.choices[0].message.content

# ─── UI ───────────────────────────────────────────
st.title("🤖 HR Policy Chatbot")
st.markdown("HR Policy பற்றி எந்த கேள்வியும் கேளுங்கள்!")

# Sidebar — PDF Upload
with st.sidebar:
    st.header("📄 PDF Upload")
    uploaded_files = st.file_uploader(
        "PDF files upload செய்யுங்கள்",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.get("processed_files", []):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = process_pdf(uploaded_file)
                    if "processed_files" not in st.session_state:
                        st.session_state.processed_files = []
                    st.session_state.processed_files.append(uploaded_file.name)
                    st.success(f"✅ {uploaded_file.name} — {chunks} chunks stored!")

    if st.session_state.get("processed_files"):
        st.divider()
        st.subheader("✅ Processed Files")
        for f in st.session_state.processed_files:
            st.write(f"📄 {f}")

    # Clear chat button
    st.divider()
    if st.button("🗑️ Chat History Clear"):
        st.session_state.messages = []
        st.rerun()

st.divider()

# Chat history initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Input
question = st.chat_input("கேள்வி கேளுங்கள்...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("யோசிக்கிறேன்..."):
            answer = get_answer(question, st.session_state.messages)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
