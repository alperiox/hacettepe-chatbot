import os

from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

import torch
from datasets import load_from_disk
from pipeline import get_embeddings
from utils import get_model, get_tokenizer


class Query(BaseModel):
    body: str

app = FastAPI()

load_dotenv()

DEFAULT_DEVICE = os.getenv("INFERENCE_DEFAULT_DEVICE")
BATCH_SIZE = os.getenv("INFERENCE_BATCH_SIZE")
TOP_K = int(os.getenv("TOP_K"))

logger.debug("Loading the model and tokenizer")
model, tokenizer = get_model(), get_tokenizer()
device = torch.device(DEFAULT_DEVICE)
logger.debug("Loaded the model and the tokenizer")

logger.debug("Initializing the dataset")
dataset = load_from_disk(os.getenv("RETRIEVAL_DATASET_PATH"))

logger.debug("Adding embeddings to FAISS index")
dataset.add_faiss_index(column="embeddings")
logger.debug("Added the embeddings as FAISS index")

def prepare_data(text: str):
    logger.debug(f"Input string:{text}") 
    embeddings = get_embeddings([text], tokenizer, model, device)
    return embeddings


@app.post("/inference")
def inference(query: Query):
    logger.debug(f"Query:{query}")
    embedding = get_embeddings(query.body, tokenizer, model, device)
    logger.debug(f"Embedding shape: {embedding.shape}")
    scores, samples = dataset.get_nearest_examples("embeddings", embedding, k=TOP_K)
    scores, texts = list(scores.astype("float")), samples['text']

    logger.debug(f"Scores:{scores}")
    
    return {"result": [
        {"text": text, "score": score} for text, score in zip(texts, scores)
        ]}

