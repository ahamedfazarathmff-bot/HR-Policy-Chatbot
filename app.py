import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, index, groq_client

model, index, groq_client = load_models()

def get_answer(question: str):
    # Embed question
    question_embedding = model.encode(question).tolist()

    # Search Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    # Get context
    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n\n"

    # Ask Groq
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an HR Policy assistant. Answer questions based only on the provided context. Be clear and concise."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content

# UI
st.title("🤖 HR Policy Chatbot")
st.markdown("HR Policy பற்றி எந்த கேள்வியும் கேளுங்கள்!")
st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Input
question = st.chat_input("கேள்வி கேளுங்கள்...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get and show answer
    with st.chat_message("assistant"):
        with st.spinner("யோசிக்கிறேன்..."):
            answer = get_answer(question)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})