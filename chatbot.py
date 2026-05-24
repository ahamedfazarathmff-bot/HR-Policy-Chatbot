from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_answer(question: str):
    # Step 1: Embed the question
    question_embedding = model.encode(question).tolist()

    # Step 2: Search Pinecone
    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    # Step 3: Get context
    context = ""
    for match in results['matches']:
        context += match['metadata']['text'] + "\n\n"

    # Step 4: Ask Groq LLM
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

# Chat loop
print("🤖 HR Policy Chatbot - கேள்வி கேளுங்கள்! ('quit' type செய்தால் exit)")
print("-" * 50)

while True:
    question = input("\nநீங்கள்: ")
    if question.lower() == "quit":
        print("Goodbye!")
        break
    answer = get_answer(question)
    print(f"\n🤖 Bot: {answer}")