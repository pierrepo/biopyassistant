"""Creates the vectorial Chroma database from Markdown files in the specified directory.

This script loads Markdown files from the specified directory, concatenates their content, 
and splits the content into chunks based on headers and word limits. The resulting chunks are saved to a ChromaDB database.

Usage:
======
    python src/create_database.py --data-path [data-path] --chroma-path [chroma-path] --chunk-size [chunk-size] --chunk-overlap [chunk-overlap] 

Arguments:
==========
    --data-path : str
        The directory containing the processed Markdown files of the python course.
    --chroma-path : str
        The name of the output path to save the ChromaDB database.
    --chunk-size : int (optional)
        The size of the text chunks to be created. Default is 1000.
    --chunk-overlap : int (optional)
        The overlap between text chunks. Default is 200.
    

Example:
========
    python src/create_database.py --data-path data/markdown_processed --chroma-path chroma_db

This command will create a vectorial Chroma database from the processed Markdown files located in the `data/markdown_processed` directory.
The text will be split into chunks of 1000 characters with an overlap of 200 characters.
And finally the vectorial Chroma database will be saved to the `chroma_db` directory.
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
import sys
import shutil
import argparse
import unicodedata

import tiktoken
from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# CONSTANTS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-large"


# FUNCTIONS
def get_args() -> tuple[str, str, int, int]:
    """Parse command-line arguments.

    Returns
    -------
    data_path, chroma_output_path, chunk_size, chunk_overlap : Tuple[str, str, int, int]
        - data_path : str
            The directory containing the processed Markdown files of the python course.
        - chroma_output_path : str
            The name of the output path to save the ChromaDB database.
        - chunk_size : int
            The size of the text chunks to be created.
        - chunk_overlap : int
            The overlap between text chunks.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Create a ChromaDB database from Markdown files in the specified directory."
    )
    # Add the arguments
    parser.add_argument(
        "-d",
        "--data-path",
        dest="data_path",
        help="The directory containing the processed Markdown files of the python course.",
    )
    parser.add_argument(
        "-c",
        "--chroma-path",
        dest="chroma_path",
        help="The name of the output path to save the ChromaDB database.",
    )
    parser.add_argument(
        "-s",
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=CHUNK_SIZE,
        help="The size of the text chunks to be created.",
    )
    parser.add_argument(
        "-o",
        "--chunk-overlap",
        dest="chunk_overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="The overlap between text chunks.",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Checks
    # db_path should exist
    if not os.path.exists(args.data_path):
        logger.error(f"The data directory '{args.data_path}' does not exist.")
        sys.exit(1)
    if args.chunk_size <= 0:
        logger.error("The chunk size should be a positive integer.")
        sys.exit(1)
    if args.chunk_overlap <= 0:
        logger.error("The chunk overlap should be a positive integer.")
        sys.exit(1)
    # chunk_overlap should be less than chunk_size
    if args.chunk_overlap >= args.chunk_size:
        logger.error(
            f"The chunk overlap ({args.chunk_overlap}) should be less than the chunk size ({args.chunk_size})."
        )
        sys.exit(1)

    return (
        args.data_path,
        args.chroma_path,
        args.chunk_size,
        args.chunk_overlap,
    )


def load_documents(data_dir: str) -> list[Document]:
    """Load Markdown documents, concatenate their content, and extract the name of the Markdown files.

    Parameters
    ----------
    data_dir : str
        The directory containing the Markdown files to be processed.

    Returns
    -------
    documents : list of Document
        List of Markdown documents.
    """
    # Load Markdown documents from the specified directory
    logger.info("Loading Markdown documents...")
    loader = DirectoryLoader(
        data_dir, glob="*.md", show_progress=True, loader_cls=TextLoader
    )
    documents = loader.load()

    logger.success("Markdown document loading complete.\n")

    return documents


def get_file_names(documents: list[Document]) -> list[str]:
    """Extract the file names of the Markdown documents.

    Parameters
    ----------
    documents : list of Document
        List of Markdown documents.

    Returns
    -------
    file_names : list of str
        List of file names of the Markdown documents.
    """
    logger.info("Extracting file names...")

    file_names = []
    for document in documents:
        # Extract the file name from the metadata source
        source = document.metadata.get("source", "")
        if source:
            file_name = source.split("/")[-1].split(".")[
                0
            ]  # Extract the file name without extension
            file_names.append(file_name)

    logger.success("Extracted file names successfully.\n")

    return sorted(file_names)


def concatenate_content(documents: list[Document]) -> str:
    """Concatenate the content of the Markdown documents.

    Parameters
    ----------
    documents : list of Document
        List of Markdown documents.

    Returns
    -------
    concatenated_content : str
        The concatenated content of all the Markdown documents.
    """
    logger.info("Concatenating content...")

    concatenated_content = ""
    for document in documents:
        # Add the document content to the concatenated content
        concatenated_content += document.page_content + "\n"
    logger.info(
        f"There is {len(concatenated_content)} characters in the concatenated content."
    )

    logger.success("Concatenated content successfully.\n")

    return concatenated_content


def split_text(content: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split concatenated Markdown content into chunks based on headers and word limits.

    Parameters:
    -----------
    content : str
        Concatenated Markdown content to be split into chunks.
    chunk_size : int
        The size of the text chunks to be created.
    chunk_overlap : int
        The overlap between text chunks.

    Returns:
    --------
    chunks : list of Document
        List of text chunks after splitting with content and metadata.
        format : [{"page_content": str, "metadata": dict}, ...]
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # split on paragraphs and sentences
        separators=["\n\n", "\n"],
    )

    # Split the resulting chunks further based on character limits
    chunks = text_splitter.split_documents(md_header_splits)

    logger.success(f"Split documents into {len(chunks)} chunks.\n")

    # tests
    # print(chunks[1], end="\n\n")
    # print(chunks[100], end="\n\n")
    # print(chunks[3000], end="\n\n")

    return chunks


def remove_small_chunks(
    chunks: list[Document], min_nb_char: int = 100
) -> list[Document]:
    """Remove small chunks from the list of text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to be filtered.
    min_nb_char : int
        Minimum number of characters for a chunk to be kept.

    Returns
    -------
    chunks : list of Document
        List of text chunks after removing small chunks.
    """
    logger.info("Removing small chunks...")
    logger.info(f"Number of chunks before removing small chunks: {len(chunks)}")

    # Remove chunks with less than min_nb_char characters
    chunks_cleaned = [
        chunk for chunk in chunks if len(chunk.page_content) >= min_nb_char
    ]

    logger.info(f"Number of chunks after removing small chunks: {len(chunks_cleaned)}")
    logger.info(f"Number of chunks removed: {len(chunks) - len(chunks_cleaned)}\n")
    logger.success("Removed small chunks successfully.\n")

    return chunks_cleaned


def add_index_to_metadata(chunks: list[Document]) -> list[Document]:
    """Add an index to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which an index is to be added.

    Returns
    -------
    chunks : list of Document
        List of text chunks with an index added to their metadata.
    """
    logger.info("Adding index to metadata...")

    # Add an index to metadata of each chunk
    for index, chunk in enumerate(chunks):
        chunk.metadata["id"] = index

    logger.success("Added index to metadata successfully.\n")

    return chunks


def add_token_number_to_metadata(chunks: list[Document]) -> list[Document]:
    """Add the number of tokens to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which the number of tokens is to be added.

    Returns
    -------
    chunks : list of Document
        List of text chunks with the number of tokens added to their metadata.
    """
    logger.info("Adding the number of tokens to metadata...")

    # Get the encoding for tokenization
    # for openai embeddings
    encoding = tiktoken.get_encoding("cl100k_base")

    # Add the number of tokens to metadata of each chunk
    for chunk in chunks:
        # Encode the chunk content
        token = encoding.encode(chunk.page_content)
        # Count the number of tokens in the chunk content
        nb_tokens = len(token)
        chunk.metadata["nb_tokens"] = nb_tokens

    logger.success("Added the number of tokens to metadata successfully.\n")

    return chunks


def add_file_names_to_metadata(
    chunks: list[Document], file_names: list[str]
) -> list[Document]:
    """Add file names to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which file names are to be added.
    file_names : list of str
        List of file names of the Markdown documents.

    Returns
    -------
    chunks : list of Document
        List of text chunks with file names added to their metadata.
    """
    logger.info("Adding file names to metadata...")

    # Add file names to metadata of each chunk
    for chunk in chunks:
        # Extract chapter_name from the metadata
        chapter_name = chunk.metadata.get("chapter_name", "")
        # Get the chapter number or appendix letter
        chapter_number = re.match(r"^\d+\s", chapter_name)
        appendix_letter = re.search(r"\b[A-Z]", chapter_name)
        # Corresponding chapter number or appendix letter with file name
        for file_name in file_names:
            if chapter_number and file_name.startswith(
                f"{chapter_number.group(0).strip().zfill(2)}_"
            ):  # zfill(2) to pad with zeros
                chunk.metadata["file_name"] = file_name
                break 
            elif appendix_letter.group(0) == file_name.split("_")[1]:
                chunk.metadata["file_name"] = file_name
                break

    logger.success("Added file names to metadata successfully.\n")

    return chunks


def preprocess_for_url(text: str, is_subsubsection: bool = False) -> str:
    """Preprocess text for creating URL.

    Parameters
    ----------
    text : str
        Text to be preprocessed.

    Returns
    -------
    str
        Processed text suitable for URL.
    """
    # Remove apostrophes
    processed_text = text.replace("'", "")

    # Remove accents
    processed_text_normalized = unicodedata.normalize("NFD", text)
    processed_text = processed_text_normalized.encode("ascii", "ignore").decode("utf-8")

    # Convert to lowercase
    processed_text = processed_text.lower()

    # Remove characters other than letters, digits, spaces, or hyphens
    processed_text = re.sub(r"[^\w\s-]", "", processed_text)

    # Replace multiple spaces with a single space
    processed_text = re.sub(r"\s+", " ", processed_text)

    # Remove points
    processed_text = processed_text.replace(".", "")

    # Replace spaces with hyphens
    processed_text = processed_text.replace(" ", "-")

    # Remove non-alphabetic characters from the end
    processed_text = re.sub(r"[^a-zA-Z]*$", "", processed_text)

    # Remove the subsubsection number
    if is_subsubsection:
        processed_text = re.sub(r"^\d+-?", "", processed_text)

    # Add a '#' at the beginning
    processed_text = "#" + processed_text

    return processed_text


def add_url_to_metadata(chunks: list[Document]) -> list[Document]:
    """Add URL to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which URL is to be added.

    Returns
    -------
    chunks : list of Document
        List of text chunks with URL added to their metadata.
    """
    logger.info("Adding URL to metadata...")

    # Add URL to metadata of each chunk
    for chunk in chunks:
        # Extract file name from the metadata
        file_name = chunk.metadata.get("file_name", "")

        # Extract section_id from the metadata
        if chunk.metadata.get("subsubsection_name", ""):  # subsubsection
            section_id = preprocess_for_url(
                chunk.metadata.get("subsubsection_name", ""), True
            )
        elif chunk.metadata.get("subsection_name", ""):  # subsection
            section_id = preprocess_for_url(chunk.metadata.get("subsection_name", ""))
        elif chunk.metadata.get("section_name", ""):  # section
            section_id = preprocess_for_url(chunk.metadata.get("section_name", ""))
        else:  # chapter
            section_id = preprocess_for_url(chunk.metadata.get("chapter_name", ""))

        # Add URL to metadata
        chunk.metadata["url"] = (
            f"https://python.sdv.univ-paris-diderot.fr/{file_name}/{section_id}"
        )

    logger.success("Added URL to metadata successfully.\n")

    return chunks


def save_to_chroma(chunks: list[Document], chroma_output_path: str) -> None:
    """Save text chunks to ChromaDB.

    Parameters
    ----------
    chunks : list of str
        List of text chunks to save to ChromaDB.
    chroma_output_path : str
        The name of the output path to save the ChromaDB database.
    """
    logger.info("Saving to Chroma...")

    # Clear out the database first.
    if os.path.exists(chroma_output_path):
        shutil.rmtree(chroma_output_path)

    # Create a new DB from the documents and save it to disk
    model_embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        chunks,
        model_embedding,
        persist_directory=chroma_output_path,
        collection_metadata={"hnsw:space": "cosine"},
    )  # distance metric

    logger.success(f"Saved {len(chunks)} chunks to {chroma_output_path}.")


def generate_data_store() -> None:
    """Generates data store by loading, splitting text into chunks, adding metadata and saving the chunks to ChromaDB."""
    # get command-line arguments
    data_path, chroma_path, chunk_size, chunk_overlap = get_args()

    # load documents from the specified directory
    documents = load_documents(data_path)

    # extract file names from the documents
    file_names = get_file_names(documents)

    # concatenate the content of the documents
    content = concatenate_content(documents)

    # split text into chunks
    chunks = split_text(content, chunk_size, chunk_overlap)

    # remove small chunks
    chunks_cleaned = remove_small_chunks(chunks, min_nb_char=100)

    # add index to the metadata
    chunks_with_index = add_index_to_metadata(chunks_cleaned)

    # add number of tokens to the metadata
    chunks_with_tokens = add_token_number_to_metadata(chunks_with_index)

    # add file names to the chunks
    chunks_with_file_names = add_file_names_to_metadata(chunks_with_tokens, file_names)

    # add URL to the chunks
    chunks_with_url = add_url_to_metadata(chunks_with_file_names)

    # save the chunks to ChromaDB
    save_to_chroma(chunks_with_url, chroma_path)


# MAIN PROGRAM
if __name__ == "__main__":
    generate_data_store()
