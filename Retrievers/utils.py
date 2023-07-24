from langchain.document_loaders import TextLoader


def write_to_file(name: str, content: str):
    # write text to local file
    with open(name, "w") as file:
        file.write(content)


def read_from_file(name: str):
    # use TextLoader to load text from local file
    loader = TextLoader(name)
    return loader.load()
