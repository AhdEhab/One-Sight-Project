import torch
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
SECRET_KEY = os.getenv("HG_KEY")
login(SECRET_KEY)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("AhmedBadawy11/multilingual-e5-small-finetuned-ar", device=device)

def embed(text):
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding