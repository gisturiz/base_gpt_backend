import os
from openai import OpenAI
import pinecone
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# openai key
client = OpenAI()

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
# find your environment next to the api key in pinecone console
env = os.getenv("PINECONE_INDEX") or "PINECONE_INDEX"
pinecone.init(api_key=api_key, enviroment=env)

# openai function without context
def complete(query, context):
    #augment the query with context
    augmented_query = "\n\n---\n\n".join(context)+"\n\n-----\n\n"+query

    res = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': f"""You are financial Q&A bot. A highly intelligent system that answers user questions based on the information provided by the user above each question. If the information can not be found in the information provided by the user you truthfully say "I don't know"."""},
            {'role': 'user', 'content': augmented_query}
        ],
        temperature=0,
    )
    return res['choices'][0]['text'].strip()


# connect to index
index_name = os.getenv("PINECONE_INDEX") or "PINECONE_INDEX"
index = pinecone.Index(index_name)

# choose embed model
embed_model = "text-embedding-ada-002"

limit = 3750
# get context from pinecone db


def retrieve(query):
    res = client.embeddings.create(
        input=[query],
        model=embed_model
    )

    # retrieve from Pinecone
    xq = res.data[0].embedding

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    print(res)
    URL = res['matches'][0]['metadata']['url']
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    return [contexts, URL]
class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    question: str
    answer: str
    url: str

# initialize app
app = FastAPI()

# enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": "0.0.1"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    query_with_contexts = retrieve(payload.text)
    res = complete(payload.text, query_with_contexts[0])
    url = query_with_contexts[1]
    return {"question": payload.text, "answer": res, "url": url}