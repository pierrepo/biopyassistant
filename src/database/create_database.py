"""Creates the ChromaDB database from Markdown files in the specified directory, 

Usage:
======
    python src/database/create_database.py [--data_dir]

Options:
    --data_dir : str, optional
        The directory containing the Markdown files to be processed. Default: processed_python_courses"


"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import re
import shutil

from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)


# CONSTANTS
CHROMA_PATH = "chroma_db"
PROCESSED_DATA_PATH = "data/processed_python_courses"


# FUNCTIONS
def load_documents() -> tuple[str, list[str]] :
    """Load Markdown documents, concatenate their content, and extract the name of the Markdown files.

    Returns
    -------
    concatenated_content, file_names : Tuple[str, List[str]]
        - concatenated_content : str
            The concatenated content of all the Markdown documents.
        - file_names : List[str]
            The list of file names of the Markdown documents.
    """
    concatenated_content = ""
    file_names = []

    # Load Markdown documents from the specified directory
    logger.info("Loading Markdown documents...")
    loader = DirectoryLoader(
        PROCESSED_DATA_PATH, glob="*.md", show_progress=True, loader_cls=TextLoader
    )
    documents = loader.load()

    for document in documents:
        # Add the document content to the concatenated content
        concatenated_content += document.page_content + "\n"
        
        # Extract the file name from the metadata source
        source = document.metadata.get('source', '')
        if source:
            file_name = source.split('/')[-1].split('.')[0]  # Extract the file name without extension
            file_names.append(file_name)

    logger.success(f"Markdown document loading complete.\n")

    return concatenated_content, file_names


def split_text(content: str) -> list[Document]:
    """Split concatenated Markdown content into chunks based on headers and word limits.

    Parameters:
    -----------
    content : str
        Concatenated Markdown content to be split into chunks.

    Returns:
    --------
    chunks : list of Document
        List of text chunks after splitting with content and metadata.
    """
    logger.info("Splitting the documents...")

    # create a Markdown header text splitter
    headers_to_split_on = [
        ("#", "chapter_name"),
        ("##", "section_name"),
        ("###", "subsection_name"),
        ("####", "subsubsection_name"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    # Split the Markdown content based on headers
    md_header_splits = markdown_splitter.split_text(content)

    # Create a character-based text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100,
                                                   # split on paragraphs, sentences, and words
                                                   separators=["\n\n","\n"," "]) 
    # Split the resulting chunks further based on character limits
    chunks = text_splitter.split_documents(md_header_splits)

    logger.success(f"Split documents into {len(chunks)} chunks.\n")

    # tests
    # print(chunks[1], end="\n\n")
    # print(chunks[60], end="\n\n")
    # print(chunks[100], end="\n\n")
    # print(chunks[160], end="\n\n")
    
    return chunks


def save_to_chroma(chunks: list[Document]) -> None:
    """Save text chunks to ChromaDB.

    Parameters
    ----------
    chunks : list of str
        List of text chunks to save to ChromaDB.

    Returns
    -------
    None
    """
    logger.info("Saving to Chroma...")

    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    model_embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma.from_documents(
        chunks,  model_embedding, persist_directory=CHROMA_PATH
    )
    db.persist() # save the database to disk
    
    logger.success(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store() -> None:
    """Generates data store by loading, splitting text into chunks, and saving the chunks to ChromaDB."""
    documents, file_names = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)
    

# MAIN PROGRAM
if __name__ == "__main__":
    generate_data_store()
