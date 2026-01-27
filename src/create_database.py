"""Creates the vectorial Chroma database from Markdown files in the specified directory.

This script loads Markdown files from the specified directory, concatenates their
content, and splits the content into chunks based on headers and word limits.
The resulting chunks are saved to a ChromaDB database.

Usage:
======
    python src/create_database.py --data-path [data-path] --chroma-path [chroma-path]
                               --chunk-size [chunk-size] --chunk-overlap [chunk-overlap]
                               --model-name [model-name] --provider-name [provider-name]

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
    --model-name : str (optional)
        Name of the embedding model to use.
        Possible choices : https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs&output_modalities=embeddings
        Default is "text-embedding-3-large".
    --provider-name : str (optional)
        Name of the embedding provider to use.
        Possible choices are "openrouter" and "openai".
        Default is "openai".


Example:
========
    python src/create_database.py --data-path data/markdown_processed \
                                    --chroma-path chroma_db \
                                    --model-name text-embedding-3-large \
                                    --provider-name openai

This command will create a Chroma vector database from the processed Markdown files
located in the `data/markdown_processed` directory. The text will be split into chunks
of 1000 characters with an overlap of 200 characters and will be embedded with the model
`text-embedding-3-large`. And finally the vector database will be saved to the
`chroma_db` directory.
"""

import os
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import click
import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from logger import create_logger


def load_documents(data_dir: str) -> list[Document]:
    """Load Markdown files, concatenate their content, and extract filenames.

    Parameters
    ----------
    data_dir : str
        The directory containing the Markdown files to be processed.

    Returns
    -------
    documents : list of Document
        List of Markdown documents.
    """
    logger.info(f"Loading Markdown documents from {data_dir}...")
    # Load Markdown documents from the specified directory
    loader = DirectoryLoader(
        data_dir, glob="*.md", show_progress=False, loader_cls=TextLoader
    )
    documents = loader.load()

    # Order the documents by the file name
    documents = sorted(documents, key=lambda x: x.metadata.get("source", ""))

    logger.success(f"Loaded {len(documents)} Markdown documents successfully.")
    return documents


def split_text_into_chunks(
    content: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split concatenated Markdown content into chunks based on headers and word limits.

    Parameters
    ----------
    content : str
        Concatenated Markdown content to be split into chunks.
    chunk_size : int
        The size of the text chunks to be created.
    chunk_overlap : int
        The overlap between text chunks.

    Returns
    -------
    chunks : list of Document
        List of text chunks after splitting with content and metadata.
        format : [{"page_content": str, "metadata": dict}, ...]
    """
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

    logger.info(f"Split documents into {len(chunks)} chunks.")
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
    # Remove chunks with less than min_nb_char characters
    chunks_cleaned = [
        chunk for chunk in chunks if len(chunk.page_content) >= min_nb_char
    ]
    logger.info(
        f"Removed {len(chunks) - len(chunks_cleaned)} small chunks \
            (less than {min_nb_char} characters)."
    )
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
    # Add an index to metadata of each chunk
    for index, chunk in enumerate(chunks):
        chunk.metadata["id"] = index

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

    logger.info(
        f"Total number of characters: \
            {sum(len(chunk.page_content) for chunk in chunks):,}"
    )
    count_tokens = sum(chunk.metadata["nb_tokens"] for chunk in chunks)
    logger.info(f"Total number of tokens: {count_tokens:,}")
    return chunks


def add_file_names_to_metadata(
    chunks: list[Document], file_name: str
) -> list[Document]:
    """Add file names to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which file names are to be added.
    file_name : str
        File name of the Markdown documents.

    Returns
    -------
    chunks : list of Document
        List of text chunks with file names added to their metadata.
    """
    # Add file names to metadata of each chunk
    for chunk in chunks:
        # Extract chapter_name from the metadata
        chapter_name = chunk.metadata.get("chapter_name", "")
        # Get the chapter number or appendix letter
        chapter_number = re.match(r"^\d+\s", chapter_name)
        appendix_letter = re.search(r"\b[A-Z]", chapter_name)
        # Corresponding chapter number or appendix letter with file name
        if (
            chapter_number
            and file_name.startswith(f"{chapter_number.group(0).strip().zfill(2)}_")
        ) or appendix_letter.group(0) == file_name.split("_")[
            1
        ]:  # zfill(2) to pad with zeros
            chunk.metadata["file_name"] = file_name
            break

    return chunks


def preprocess_for_url(text: str, *, is_subsubsection: bool = False) -> str:
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

    # Remove pattern {.unnumbered}
    processed_text = re.sub(r"{.unnumbered}", "", processed_text)

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
        processed_text = re.sub(r"^[a-zA-Z]?\d+-?", "", processed_text)

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
    # Add URL to metadata of each chunk
    for chunk in chunks:
        # Extract file name from the metadata
        file_name = chunk.metadata.get("file_name", "")

        # Extract section_id from the metadata
        if chunk.metadata.get("subsubsection_name", ""):  # subsubsection
            section_id = preprocess_for_url(
                chunk.metadata.get("subsubsection_name", ""), is_subsection=True
            )
        elif chunk.metadata.get("subsection_name", ""):  # subsection
            section_id = preprocess_for_url(
                chunk.metadata.get("subsection_name", ""), is_subsection=True
            )
        elif chunk.metadata.get("section_name", ""):  # section
            section_id = preprocess_for_url(chunk.metadata.get("section_name", ""))
        else:  # chapter
            section_id = preprocess_for_url(chunk.metadata.get("chapter_name", ""))

        # Add URL to metadata
        chunk.metadata["url"] = (
            f"https://python.sdv.u-paris.fr/{file_name}/{section_id}"
        )
        logger.debug(f"Added URL: {chunk.metadata['url']}")

    return chunks


def save_to_chroma(
    chunks: list[Document],
    model_name: str,
    provider_name: str,
    chroma_output_path: Path,
) -> None:
    """Save text chunks to ChromaDB.

    Parameters
    ----------
    chunks : list[Document]
        List of text chunks to save to ChromaDB.
    model_name : str
        Name of the embedding model to use.
    provider_name : str
        Name of the embedding provider to use.
    chroma_output_path : Path
        Path to the directory where the ChromaDB database will be saved.
    """
    logger.info("Saving to ChromaDB...")
    # Clear the existing database if it exists
    if chroma_output_path.exists():
        shutil.rmtree(chroma_output_path)

    # Get the API key and base URL based on the provider
    if provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None

    # Create the embeddings for each chunks
    embeddings = OpenAIEmbeddings(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )

    # Create a ChromaDB with the embeddings
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_output_path),
        collection_metadata={"hnsw:space": "cosine"},  # distance metric
    )
    logger.success(f"Saved {len(chunks)} chunks to {chroma_output_path} successfully!")


@click.command()
@click.option(
    "-d",
    "--data-path",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing the processed Markdown files of the Python course.",
)
@click.option(
    "-c",
    "--chroma-path",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Output path to save the ChromaDB database.",
)
@click.option(
    "-s",
    "--chunk-size",
    default=1000,
    type=click.IntRange(min=1),
    help="Size of the text chunks to be created.",
)
@click.option(
    "-o",
    "--chunk-overlap",
    default=200,
    type=click.IntRange(min=0),
    help="Overlap between text chunks.",
)
@click.option(
    "-m",
    "--model-name",
    default="text-embedding-3-large",
    type=str,
    help="Name of the embedding model, chosen from OpenRouter`s available models.",
)
@click.option(
    "-p",
    "--provider-name",
    default="openrouter",
    type=str,
    help="Name of the embedding provider to use.",
)
def generate_data_store(
    data_path: Path,
    chroma_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    provider_name: str,
) -> None:
    """Build a ChromaDB store from chunked text with metadata."""
    # Set-up the logger
    log_path = f"logs/create_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = create_logger(log_path)
    logger.info("Creating Chroma database...")

    # Validate the CLI arguments
    if chunk_overlap >= chunk_size:
        logger.error(
            f"The chunk overlap ({chunk_overlap}) must be less than \
                the chunk size ({chunk_size})."
        )
        sys.exit(1)

    # Load the environment variables with LLM api keys
    load_dotenv()

    all_chunks = []
    # load documents from the specified directory
    documents = load_documents(data_path)
    logger.info("Processing files...")
    for doc in documents:
        file_name = doc.metadata.get("source", "")
        logger.info(f"File path: {file_name}")
        # split text of the current document into chunks
        chunks = split_text_into_chunks(doc.page_content, chunk_size, chunk_overlap)

        # remove small chunks
        chunks_cleaned = remove_small_chunks(chunks, min_nb_char=100)

        # add index to the metadata
        chunks_with_index = add_index_to_metadata(chunks_cleaned)

        # add number of tokens to the metadata
        chunks_with_tokens = add_token_number_to_metadata(chunks_with_index)

        # add file name to the chunk metadata
        chunks_with_file_name = add_file_names_to_metadata(
            chunks_with_tokens, file_name
        )
        # add URL to the chunk metadata if available
        chunks_with_url = add_url_to_metadata(chunks_with_file_name)

        all_chunks.extend(chunks_with_url)
        logger.info(f"Example chunk metadata: {all_chunks[-1].metadata}")

    logger.info("Summary:")
    logger.info(f"Total number of files processed: {len(documents)}")
    logger.info(f"Total number of chunks: {len(all_chunks)}")
    logger.info(
        f"Total number of characters: {
            sum(len(chunk.page_content) for chunk in all_chunks):,}"
    )
    count_tokens = sum(chunk.metadata["nb_tokens"] for chunk in all_chunks)
    logger.info(f"Total number of tokens: {count_tokens:,}")
    # save the embeddings to ChromaDB
    save_to_chroma(all_chunks, model_name, provider_name, chroma_path)


# MAIN PROGRAM
if __name__ == "__main__":
    generate_data_store()
