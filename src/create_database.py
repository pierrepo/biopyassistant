"""Creates the ChromaDB database from Markdown files in the 'data' directory.

Usage:
======
    python create_database.py

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

from langchain_openai import OpenAIEmbeddings
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
def main():
    """Main function to generate the ChromaDB database."""
    generate_data_store()


def generate_data_store():
    global number_of_docs
    """Generates data store by loading and cleaning documents,
    splitting text, and saving to ChromaDB."""
    documents, number_of_docs = load_documents()
    documents_cleaned = remove_comments_in_pythoncode(documents)
    chunks = split_text(documents_cleaned)
    save_to_chroma(chunks)


def load_documents():
    """Load Markdown documents and concatenates their content.

    Returns:
        tuple: A tuple containing the concatenated content (str) of all Markdown
            documents and the number of loaded documents.
    """
    concatenated_content = ""
    # Load Markdown documents from the specified directory
    print("\033[96mLoading Markdown documents...\033[0m")
    loader = DirectoryLoader(
        DATA_PATH, glob="*.md", show_progress=True, loader_cls=TextLoader
    )
    documents = loader.load()
    number_of_docs = len(documents)
    for document in documents:
        # Add the document content to the concatenated content
        concatenated_content += document.page_content + "\n"
        print(f"Loaded document: {document.metadata['source']}")

    print("\033[92mMarkdown document loading complete.\033[0m", end=" \n\n")
    # print(concatenated_content)
    return concatenated_content, number_of_docs


def remove_comments_in_pythoncode(content: str):
    """Remove comments from Python code blocks in Markdown content.

    Args:
        content (str): Content of Markdown file.

    Returns:
        str: Markdown content with comments removed from Python code blocks.
    """
    in_python_block = False
    cleaned_content = []
    removed_lines = 0
    print("\033[96mCleaning the documents...\033[0m")
    for line in content.split("\n"):
        if line.strip().startswith("```python"):
            in_python_block = True
            cleaned_content.append(line)
        elif line.strip().startswith("```") and in_python_block:
            in_python_block = False
            cleaned_content.append(line)
        elif in_python_block:
            line = re.sub(r"#.*", "", line)
            cleaned_content.append(line)
            removed_lines += 1
        else:
            cleaned_content.append(line)

    print(f"\033[92mNumber of comment lines removed: {removed_lines}\033[0m")
    print(
        f"\033[92mNumber of final content lines : {len(cleaned_content)}\033[0m",
        end=" \n\n",
    )

    return "\n".join(cleaned_content)


def split_text(content: str):
    """Split concatenated Markdown content into chunks based on headers and character limits.

    Parameters:
    -----------
    content : str
        Concatenated Markdown content to be split into chunks.

    Returns:
    --------
    list of str
        List of text chunks after splitting.
    """
    print("\033[96mSplitting the documents...\033[0m")
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    # Split the resulting chunks further based on character limits
    chunks = text_splitter.split_documents(md_header_splits)
    print(
        f"\033[92mSplit {number_of_docs} documents into {len(chunks)} chunks.\033[0m",
        end=" \n\n",
    )

    # tests
    # print(chunks[1], end="\n\n")
    # print(chunks[60], end="\n\n")
    # print(chunks[100], end="\n\n")
    # print(chunks[160], end="\n\n")
    return chunks


def save_to_chroma(chunks: list[str]):
    """Save text chunks to ChromaDB."""
    print("\033[96mSaving to Chroma...\033[0m")
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"\033[92mSaved {len(chunks)} chunks to {CHROMA_PATH}.\033[0m")


# MAIN PROGRAM
if __name__ == "__main__":
    main()
