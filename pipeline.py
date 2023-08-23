import json
import os
from dotenv import load_dotenv

import typer
from loguru import logger
from typer import Argument
from typing_extensions import Annotated


import pandas as pd
from datasets import Dataset
from langdetect import detect

import torch
from utils import get_device, get_model, get_tokenizer


load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
CHECKPOINT = os.getenv("CHECKPOINT")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE")
RETRIEVAL_DATASET_PATH = os.getenv("RETRIEVAL_DATASET_PATH")

app = typer.Typer()


def prepare_dataset() -> Dataset:
    def load_dataset() -> dict:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def merge(sample: dict) -> str:
        """merges the paragraphs by including the department at the start of the text, if it exists"""
        department = sample["department"]
        paragraphs = sample["paragraphs"]

        if department:
            paragraphs = [department] + paragraphs

        text_sample = "\n".join(paragraphs)

        return text_sample

    def construct(dataset: dict) -> pd.DataFrame:
        from tqdm.auto import tqdm

        _dataset = []
        base_url = dataset["base_url"]
        samples = dataset["data"]

        for s in tqdm(samples):
            sample = {"url": s["url"], "base_url": base_url, "text": merge(s)}
            _dataset.append(sample)

        df = pd.DataFrame(_dataset, columns=["url", "base_url", "text"])

        return df

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df[(df.text.str.len() > 0)]
        df.drop_duplicates(subset=["text"], inplace=True)

        return df

    def detect_languages(dataset: pd.DataFrame) -> pd.DataFrame:
        dataset["language"] = dataset.text.apply(detect)
        return dataset

    dataset = load_dataset()
    df = clean(construct(dataset))
    df = detect_languages(df)
    df = df[df.language == "tr"].reset_index(drop=True)
    dataset = Dataset.from_pandas(df)

    return dataset


def get_embeddings(text, tokenizer, model, device):
    tokenized = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    outputs = model(**tokenized)
    embeddings = outputs.last_hidden_state[:, 0].detach().cpu().numpy()

    return embeddings


@app.command()
def run(
    device: Annotated[
        str,
        Argument(
            help="Default device for the embedding calculation, tries to use GPU by default and fallbacks to cpu"
        ),
    ] = DEFAULT_DEVICE,
    batch_size: Annotated[
        int,
        Argument(
            help="Batch size for the batched processing while calculating the embeddings."
        ),
    ] = BATCH_SIZE,
    retrieval_dataset_path: Annotated[
        str, Argument(help="Default path for the resulted dataset.")
    ] = RETRIEVAL_DATASET_PATH,
):
    if device == "auto":
        device = get_device()
    else:
        device = torch.device(device)
    logger.debug(f"Running on {device}")

    logger.debug("Loading model and tokenizer")
    model = get_model()
    tokenizer = get_tokenizer()
    logger.debug("Model and tokenizer loaded")

    logger.debug("Preparing dataset")
    dataset = prepare_dataset()
    logger.debug("Dataset prepared")

    logger.debug("Getting embeddings")
    if batch_size:
        dataset = dataset.map(
            lambda x: {
                "embeddings": get_embeddings(x["text"], tokenizer, model, device)
            },
            batched=True,
            batch_size=batch_size,
        )
    else:
        dataset = dataset.map(
            lambda x: {
                "embeddings": get_embeddings(x["text"], tokenizer, model, device)
            }
        )
    logger.debug("Embeddings obtained")

    logger.debug("Saving dataset")
    dataset.save_to_disk(retrieval_dataset_path)
    logger.debug("Dataset saved")


if __name__ == "__main__":
    app()
