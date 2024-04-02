"""Creates the ChromaDB database from Markdown files in the 'data' directory.

Usage:
======
    python src/create_database.py

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
CHROMA_PATH = "chroma"
DATA_PATH = "data/python_courses"


# FUNCTIONS
def load_documents() -> str :
    """Load Markdown documents and concatenates their content.

    Returns
    -------
    str
        The concatenated content of all Markdown documents.
    """
    concatenated_content = ""
    # Load Markdown documents from the specified directory
    logger.info("Loading Markdown documents...")
    loader = DirectoryLoader(
        DATA_PATH, glob="*.md", show_progress=True, loader_cls=TextLoader
    )
    documents = loader.load()
    for document in documents:
        # Add the document content to the concatenated content
        concatenated_content += document.page_content + "\n"

    logger.success(f"Markdown document loading complete.\n")

    return concatenated_content


def clean_python_comments(content: str) -> str:
    """Remove spaces between '#' and comments in Python code blocks in Markdown content.

    Parameters
    ----------
    content : str
        Content of Markdown file.

    Returns
    -------
    str
        Markdown content with spaces removed between '#' and comments in Python code blocks.
    """
    in_python_block = False
    cleaned_content = []
    modified_comment_lines = 0

    logger.info("Cleaning the documents...")
    
    for line in content.split("\n"):
            if line.strip().startswith("```python"):
                in_python_block = True
                cleaned_content.append(line)
            elif line.strip().startswith("```") and in_python_block:
                in_python_block = False
                cleaned_content.append(line)
            elif in_python_block:
                # Check if line has '#' followed by space(s) and remove them
                modified_line = re.sub(r"#\s+", "#", line)
                if modified_line != line:  # Check if any modification was made
                    modified_comment_lines += 1
                    cleaned_content.append(modified_line)
                else:
                    cleaned_content.append(line)
                
            else:
                cleaned_content.append(line)

    logger.info(f"Number of space lines modified comment lines: {modified_comment_lines}")
    logger.info(f"Number of final content lines: {len(cleaned_content)}")
    logger.success(f"Cleaning the documents complete.\n")

    return "\n".join(cleaned_content)


def split_text(content: str) -> list[Document]:
    """Split concatenated Markdown content into chunks based on headers and word limits.

    Parameters:
    -----------
    content : str
        Concatenated Markdown content to be split into chunks.

    Returns:
    --------
    list of str
        List of text chunks after splitting.
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
                                                   separators=["\n\n","\n"," "])
    # Split the resulting chunks further based on character limits
    chunks = text_splitter.split_documents(md_header_splits)


    logger.success(f"Split documents into {len(chunks)} chunks.\n")

    # tests
    # print(chunks[1], end="\n\n")
    # print(chunks[60], end="\n\n")
    # print(chunks[100], end="\n\n")
    # print(chunks[160], end="\n\n")
    """
    # Write chunks to a file
    output_file_path = "chunks.txt"
    with open(output_file_path, "w", encoding="utf-8") as file:
        for index, chunk in enumerate(chunks):
            file.write(f"Chunk {index + 1}:\n")
            file.write("Metadata:\n")
            for key, value in chunk.metadata.items():
                file.write(f"    {key}: {value}\n")
            file.write("Content:\n")
            file.write(chunk.page_content)
            file.write("\n\n")
    """
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
    db.persist()
    
    logger.success(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store() -> None:
    """Generates data store by loading and cleaning documents,
    splitting text, and saving to ChromaDB."""
    documents = load_documents()
    documents_cleaned = clean_python_comments(documents)
    chunks = split_text(documents_cleaned)
    save_to_chroma(chunks)


# MAIN PROGRAM
if __name__ == "__main__":
    generate_data_store()
