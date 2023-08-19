from crawler import Crawler
import typer

import os

app = typer.Typer()


@app.command()
def crawl(base_url: str, max_depth: int = 1, auto_save: int = 100):
    crawler = Crawler(base_url, max_depth, auto_save)
    crawler.crawl()

@app.command()
def auto(base_url: str, max_depth: int = 1, auto_save: int = 100, load: bool = True):
    if load:
        assert os.path.exists("dataset.json"), "No dataset.json found, please run the default crawler first to save an initial dataset"
        crawler = Crawler()
        crawler.load_json()
        crawler.max_depth = max_depth
        crawler.auto_save = auto_save
        
        crawler.crawl()
    else:
        crawler = Crawler(base_url, max_depth, auto_save)
        crawler.crawl()

if __name__ == "__main__":
    app()