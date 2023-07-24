from langchain.document_loaders import SeleniumURLLoader

def scrape(urls: list[str]):
    # use the selenium scraper to load the documents
    loader = SeleniumURLLoader(urls=urls)
    return loader.load()
