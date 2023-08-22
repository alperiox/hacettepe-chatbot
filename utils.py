import os
import pandas as pd

import json
from langdetect import detect
from dotenv import load_dotenv

import torch

from transformers import AutoModel, AutoTokenizer

from loguru import logger


load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
checkpoint = os.getenv("CHECKPOINT")


def get_device():
    # mps enables gpu calculations in macbooks
    if torch.backends.mps.is_available():
        logger.debug("MPS is available for macOS GPU usage")
        if torch.backends.mps.is_built():
            logger.debug("MPS is built")
            device = torch.device("mps")
        else:
            logger.debug("MPS is not built, falling back to cpu")
            device = torch.device("cpu")

    else:
        logger.debug("MPS is not built")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device


def get_model():
    model = AutoModel.from_pretrained(checkpoint)
    device = get_device()
    model.to(device)
    return model


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer
