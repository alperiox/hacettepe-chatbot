import httpx
from bs4 import BeautifulSoup

from datetime import datetime

from dataclasses import dataclass
import json

from urllib.parse import urlparse

from loguru import logger

import os

import unicodedata

import re


@dataclass
class Url:
    """Dataclass to represent a url"""

    url: str
    history: list[str]

    def __eq__(self, other) -> bool:
        return self.url == other.url

    def __hash__(self) -> int:
        return hash(self.url)


class Crawler(object):
    def __init__(
        self,
        base_url: str = None,
        max_depth: int = 1,
        auto_save: int = 100,
        save_logs=True,
    ):
        self.save_logs = save_logs

        self.base_url = base_url
        self.visited_urls = set()
        self.target_urls = list()
        self.max_depth = max_depth
        self.department = None

        self.target_urls.append(Url(base_url, []))

        if not base_url:
            logger.debug("No base url provided, loading from dataset.json")
            self.load_json()

        self.auto_save = auto_save
        logger.debug("Initialized crawler with base_url: {}", self.base_url)
        logger.debug("Max depth: {}", self.max_depth)
        logger.debug("Auto save: {}", self.auto_save)

        self.dataset = {
            "data": [],
            "base_url": base_url,
            "max_depth": max_depth,
            "date": datetime.now().strftime("%Y-%m-%d %H:%m"),
            "autosave": self.auto_save,
        }

        self.get_department()

    def __str__(self):
        return f"Crawler(base_url={self.base_url}, max_depth={self.max_depth})"

    def __repr__(self) -> str:
        return self.__str__()

    def dump_json(self) -> None:
        """Dumps the dataset into a json file"""
        copy_dict = self.dataset.copy()

        copy_dict["visited_urls"] = [url.url for url in self.visited_urls]
        copy_dict["target_urls"] = [url.url for url in self.target_urls]

        with open("dataset.json", "w", encoding="utf-8") as f:
            json.dump(copy_dict, f, indent=4, ensure_ascii=False)

    def load_json(self) -> None:
        """Loads the dataset from a json file"""
        with open("dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dataset = data
        self.base_url = data["base_url"]
        self.max_depth = data["max_depth"]

        self.visited_urls = set(
            [Url(sample["url"], sample["history"]) for sample in data["data"]]
        )
        self.target_urls = set([Url(url, []) for url in data["target_urls"]])

    def fix_url(self, url: str, base_url: str) -> str:
        """Fixes the url to be absolute"""

        if url.startswith("/"):
            return os.path.join(base_url, url[1:])
        elif url.startswith("http"):
            return url
        else:
            return os.path.join(base_url, url)

    def text_preprocessor(self, text: str) -> str:
        """Preprocesses the text"""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\n+", "\n", text)

        return unicodedata.normalize("NFKC", text.strip())

    def _get_soup(self, url: Url) -> BeautifulSoup:
        try:
            response = httpx.get(url.url, follow_redirects=True)
            response.raise_for_status()
        except Exception:
            logger.error("Error while crawling url: {}", url.url)
            self.visited_urls.add(url)
            return None
        soup = BeautifulSoup(response.text, "html.parser")

        return soup

    def get_department(self) -> None:
        soup = self._get_soup(Url(self.base_url, []))
        department = soup.find("div", {"class": "banner_uni_bolum"})
        self.department = department.text if department else None

    def get_data(self, url: Url) -> dict[str, list]:
        """Extracts the html and anchor tags from the given url"""

        soup = self._get_soup(url)
        if not soup:
            return None

        # extract the anchor tags
        anchor_tags = soup.find_all("a")

        if urlparse(self.base_url).netloc in url.url and self.get_department:
            department = self.department
        else:
            department = soup.find("div", {"class": "banner_uni_bolum"})
            department = department.text if department else None

        anchor_urls = [
            self.fix_url(tag.get("href"), url.url)
            for tag in anchor_tags
            if tag.get("href") is not None
        ]
        # extract the paragraph tags
        paragraphs = soup.find_all("p")
        # preprocess the paragraphs
        paragraphs = [self.text_preprocessor(p.text) for p in paragraphs]

        self.visited_urls.add(url)

        return {
            "anchor_urls": anchor_urls,
            "paragraphs": paragraphs,
            "url": url.url,
            "history": url.history + [url.url],
            "department": department,
        }

    def crawl(self) -> None:
        """Crawls through the target url and extracts the data"""
        if self.save_logs:
            logger.add(
                "dataset_errors_{time}.log",
                rotation="500MB",
                compression="zip",
                level="ERROR",
            )
            logger.add(
                "dataset_{time}.log", rotation="500MB", compression="zip", level="DEBUG"
            )

        c = 0

        for url in self.target_urls:
            if len(url.history) >= self.max_depth:
                logger.debug("Max depth reached for url: {}", url.url)
                continue

            logger.debug("Crawling url: {}", url.url)

            data = self.get_data(url)
            if not data:
                continue

            self.dataset["data"].append(data)

            logger.debug("Found {} anchor urls", len(data["anchor_urls"]))
            logger.debug("Found {} paragraphs", len(data["paragraphs"]))

            for href in data["anchor_urls"]:
                href = Url(href, url.history + [url.url])
                if (href not in self.visited_urls) and (href not in self.target_urls):
                    self.target_urls.append(href)

            self.target_urls.remove(url)

            c += 1
            if self.auto_save % c == 0 and self.auto_save:
                self.dump_json()
                logger.info(
                    "Auto saved dataset: {} | number of target urls: {} | number of processed urls: {}",
                    len(self.dataset["data"]),
                    len(self.target_urls),
                    len(self.visited_urls),
                )

        self.dump_json()
        logger.info(
            "Finished crawling | number of target urls: {} | number of processed urls: {}",
            len(self.target_urls),
            len(self.visited_urls),
        )
