"""Creates the vectorial Chroma database from Markdown files in the specified directory.

This script loads Markdown files from the specified directory, concatenates their
content, and splits the content into chunks based on headers and word limits.
The resulting chunks are saved to a ChromaDB database.

Usage:
======
    python src/create_database.py --course-yaml [course-yaml]
                               --chroma-path [chroma-path]
                               --chunk-size [chunk-size] --chunk-overlap [chunk-overlap]
                               --model-name [model-name] --provider-name [provider-name]

Arguments:
==========
    --course-yaml : str
        The YAML file containing the course structure, including chapter names, titles,
        source Markdown paths, and processed file paths.
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
    python src/create_database.py --course-yaml data/chapters_and_levels.yaml \
                                    --chroma-path chroma_db \
                                    --model-name text-embedding-3-large \
                                    --provider-name openai

This command will create a Chroma vector database from the processed Markdown files
located in the paths specified in the `data/chapters_and_levels.yaml` file.
The text will be split into chunks of 1000 characters with an overlap of 200 characters
and will be embedded with the model `text-embedding-3-large`.
And finally the vector database will be saved to the `chroma_db` directory.
"""

import os
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import click
import loguru
import tiktoken
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI

from logger import create_logger
from parse_clean_markdown import load_chapters_from_yaml


class OpenRouterEmbeddings(Embeddings):
    """LangChain-compatible embeddings using OpenRouter."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        logger: "loguru.Logger" = loguru.logger,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.logger = logger
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed search query.

        Returns
        -------
            Embedding for the input query text.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def embed_documents(
        self,
        texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]:
        """Embed documents one by one with retries and chunk logging.

        Parameters
        ----------
        texts : list[str]
            List of input texts to embed.
        max_retries : int
            Maximum number of retry attempts for a failed embedding.

        Returns
        -------
        list[list[float]]
            List of embeddings, one per input text.
        """
        embeddings: list[list[float]] = []

        for idx, text in enumerate(texts):
            for attempt in range(1, max_retries + 1):
                try:
                    embedding = self.embed_query(text=text)
                    embeddings.append(embedding)
                    self.logger.debug(
                        f"Chunk {idx} embedded successfully on attempt {attempt}."
                    )
                    break  # Exit the retry loop on success
                except Exception as e:
                    self.logger.warning(
                        f"Attempt {attempt} failed for chunk {idx}: {e}"
                    )
                    if attempt == max_retries:
                        self.logger.error(
                            f"Failed to embed chunk {idx} after {max_retries} attempts."
                        )
                        raise  # Re-raise the exception after max retries

        return embeddings


def load_documents(
    course_yaml: Path, logger: "loguru.Logger" = loguru.logger
) -> list[Document]:
    """Load Markdown files, concatenate their content, and extract filenames.

    Parameters
    ----------
    course_yaml : Path
        The YAML file containing the course structure, including chapter names, titles,
        source Markdown paths, and processed file paths.
    logger: "loguru.Logger"
        Logger for logging messages.

    Returns
    -------
    documents : list of Document
        List of Markdown documents.
    """
    documents = []

    # Load chapters and levels from the YAML file
    chapters = load_chapters_from_yaml(course_yaml, logger)

    # Iterate through the chapters and load the processed Markdown files as Documents
    logger.info("Converting Markdown file to Document...")
    for chapter in chapters:
        # Get the processed Markdown file path
        processed_path = chapter.get("processed_file_path")
        # Check if the processed file path is defined for the chapter
        if not processed_path:
            logger.warning(
                f"No processed_file_path defined for chapter id={chapter.get('id')} "
                f"title={chapter.get('title')}"
            )
            continue

        processed_path = Path(processed_path)
        # Check if the processed Markdown file exists
        if not processed_path.exists():
            logger.warning(f"Processed Markdown file not found: {processed_path}")
            continue
        # Load the processed Markdown file as a Document
        loader = TextLoader(processed_path)
        doc = loader.load()
        documents.extend(doc)
        logger.debug(
            f"Converted {processed_path} to Document with "
            f"{len(doc[0].page_content)} characters."
        )

    # Order the documents by the file name
    documents = sorted(documents, key=lambda x: x.metadata.get("source", ""))

    logger.success(
        f"Converted {len(documents)}/{len(chapters)} Markdown documents successfully."
    )
    return documents


def split_text_into_chunks(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    logger: "loguru.Logger" = loguru.logger,
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
    logger: "loguru.Logger"
        Logger for logging messages.

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
    chunks: list[Document],
    logger: "loguru.Logger" = loguru.logger,
    min_nb_char: int = 100,
) -> list[Document]:
    """Remove small chunks from the list of text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to be filtered.
    min_nb_char : int
        Minimum number of characters for a chunk to be kept.
    logger: "loguru.Logger"
        Logger for logging messages.

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
        f"Removed {len(chunks) - len(chunks_cleaned)} small chunks"
        f" (less than {min_nb_char} characters)."
    )
    return chunks_cleaned


def add_token_number_to_metadata(
    chunks: list[Document],
    logger: "loguru.Logger" = loguru.logger,
) -> list[Document]:
    """Add the number of tokens to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which the number of tokens is to be added.
    logger: "loguru.Logger"
        Logger for logging messages.

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
    chunks: list[Document], file_path: str, logger: "loguru.Logger" = loguru.logger
) -> list[Document]:
    """Add file paths to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which file paths are to be added.
    file_path : str
        File path of the Markdown documents.
    logger: "loguru.Logger"
        Logger for logging messages.

    Returns
    -------
    chunks : list of Document
        List of text chunks with file paths added to their metadata.
    """
    file_name = Path(file_path).name

    # Determine chapter id based on the file name
    chapter_id = None
    # Match numbered chapters, e.g., "01_intro.md" or "24_avoir_plus_la_classe.md"
    match_chapter = re.match(r"(\d+)_", file_name)
    if match_chapter:
        chapter_id = str(match_chapter.group(1))
    else:
        # Match annexes, e.g., "annexe_A.md"
        match_annex = re.match(r"annexe[_-]([A-Z0-9]+)", file_name, re.IGNORECASE)
        if match_annex:
            chapter_id = match_annex.group(1).upper()

    if chapter_id is None:
        logger.warning(
            f"Could not extract chapter index from file name: {file_name}. "
            "No chapter index added to metadata."
        )

    # Add metadata to each chunk
    for chunk in chunks:
        chunk.metadata["file_path"] = file_path
        chunk.metadata["file_name"] = Path(file_path).name
        chunk.metadata["chapter_id"] = chapter_id

    return chunks


def preprocess_for_url(text: str, *, is_subsubsection_name: bool = False) -> str:
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
    if is_subsubsection_name:
        processed_text = re.sub(r"^[a-zA-Z]?\d+-?", "", processed_text)

    # Add a '#' at the beginning
    processed_text = "#" + processed_text

    return processed_text


def add_url_to_metadata(
    chunks: list[Document], logger: "loguru.Logger" = loguru.logger
) -> list[Document]:
    """Add URL to the metadata of the text chunks.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to which URL is to be added.
    logger: "loguru.Logger"
        Logger for logging messages.

    Returns
    -------
    chunks : list of Document
        List of text chunks with URL added to their metadata.
    """
    # Add URL to metadata of each chunk
    for i, chunk in enumerate(chunks, start=1):
        # Extract file name from the metadata
        file_path = chunk.metadata.get("file_name", "")
        file_name = Path(file_path).stem

        # Extract section_id from the metadata
        if chunk.metadata.get("subsubsection_name", ""):  # subsubsection
            section_id = preprocess_for_url(
                chunk.metadata.get("subsubsection_name", ""), is_subsubsection_name=True
            )
        elif chunk.metadata.get("subsection_name", ""):  # subsection
            section_id = preprocess_for_url(chunk.metadata.get("subsection_name", ""))
        elif chunk.metadata.get("section_name", ""):  # section
            section_id = preprocess_for_url(chunk.metadata.get("section_name", ""))
        else:  # chapter
            section_id = preprocess_for_url(chunk.metadata.get("chapter_name", ""))

        # Add URL to metadata
        chunk.metadata["url"] = (
            f"https://python.sdv.u-paris.fr/{file_name}/{section_id}"
        )
        logger.debug(f"Chunk {i} URL: {chunk.metadata['url']}")

    return chunks


def create_embeddings_function(
    model_name: str,
    provider_name: str,
) -> OpenAIEmbeddings | OpenRouterEmbeddings:
    """Create an embeddings function based on the specified model and provider.

    Parameters
    ----------
    model_name : str
        Name of the embedding model to use.
    provider_name : str
        Name of the embedding provider to use.

    Returns
    -------
    OpenAIEmbeddings | OpenRouterEmbeddings
        An instance of the OpenAIEmbeddings or
        OpenRouterEmbeddings class initialized
        with the specified model and provider.

    Raises
    ------
    KeyError
        If the API key for the specified provider is not found in environment variables.
    """
    # Load the environment variables with LLM api keys
    load_dotenv()
    try:
        # Get the API key and base URL based on the provider
        if provider_name == "openrouter":
            return OpenRouterEmbeddings(
                model=model_name,
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
        elif provider_name == "openai":
            return OpenAIEmbeddings(
                model=model_name, api_key=os.getenv("OPENAI_API_KEY")
            )
    # Handle the case where the API key is not found in environment variables
    except KeyError as e:
        msg = f"API key for {provider_name} not found in environment variables."
        raise KeyError(msg) from e


def save_to_chroma(
    chunks: list[Document],
    model_name: str,
    provider_name: str,
    chroma_output_path: Path,
    logger: "loguru.Logger" = loguru.logger,
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
    logger: "loguru.Logger"
        Logger for logging messages.
    """
    logger.info(f"Saving to ChromaDB using embedding model `{model_name}`...")
    # Clear the existing database if it exists
    if chroma_output_path.exists():
        shutil.rmtree(chroma_output_path)

    # Create the embeddings function
    embeddings_function = create_embeddings_function(model_name, provider_name)

    # Generate unique IDs for each chunk
    uuids = [str(index) for index in range(len(chunks))]

    # Create a ChromaDB with the embeddings
    Chroma.from_documents(
        documents=chunks,
        ids=uuids,
        embedding=embeddings_function,
        persist_directory=str(chroma_output_path),
        collection_metadata={"hnsw:space": "cosine"},  # distance metric
    )
    logger.success(f"Saved {len(chunks)} chunks to {chroma_output_path} successfully!")


@click.command()
@click.option(
    "--course-yaml",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to the YAML file defining the course chapters and student levels. "
        "The YAML should include chapter names, titles, source Markdown paths, "
        "and processed file paths."
    ),
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
    type=click.IntRange(min=50),
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
    help="Name of the embedding model,"
    "chosen from OpenAI and OpenRouter`s embedding models. ",
)
@click.option(
    "-p",
    "--provider-name",
    default="openai",
    type=click.Choice(["openrouter", "openai"], case_sensitive=False),
    help="Name of the embedding provider to use.",
)
def generate_data_store(
    course_yaml: Path,
    chroma_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    provider_name: str,
) -> None:
    """Build a ChromaDB store from chunked text with metadata."""
    # Set-up the logger
    log_path = f"logs/{datetime.now().strftime('%Y%m%d')}/create_database.log"
    logger = create_logger(log_path)
    logger.info("Creating Chroma database...")

    # Validate the CLI arguments
    if chunk_overlap >= chunk_size:
        logger.error(
            f"The chunk overlap ({chunk_overlap}) must be less than \
                the chunk size ({chunk_size})."
        )
        sys.exit(1)

    all_chunks = []
    # load documents from the specified directory
    documents = load_documents(course_yaml, logger)
    logger.info("Processing Documents...")
    for doc in documents:
        file_name = doc.metadata.get("source", "")
        logger.info(f"File path: {file_name}")
        # split text of the current document into chunks
        chunks = split_text_into_chunks(
            doc.page_content, chunk_size, chunk_overlap, logger
        )
        # remove small chunks
        chunks_cleaned = remove_small_chunks(chunks, logger, min_nb_char=100)

        # add number of tokens to the metadata
        chunks_with_tokens = add_token_number_to_metadata(chunks_cleaned, logger)

        # add file name and chapter id to the chunk metadata
        chunks_with_file_name = add_file_names_to_metadata(
            chunks_with_tokens, file_name, logger
        )
        # add URL to the chunk metadata if available
        chunks_with_url = add_url_to_metadata(chunks_with_file_name, logger)

        all_chunks.extend(chunks_with_url)
        logger.debug(f"Example chunk metadata: {all_chunks[-1].metadata}")

    logger.info("Summary:")
    logger.info(f"Total number of files processed: {len(documents)}")
    logger.info(f"Total number of chunks: {len(all_chunks)}")
    logger.info(
        f"Total number of characters: \
        {sum(len(chunk.page_content) for chunk in all_chunks):,}"
    )
    count_tokens = sum(chunk.metadata["nb_tokens"] for chunk in all_chunks)
    logger.info(f"Total number of tokens: {count_tokens:,}")

    # save the embeddings to ChromaDB
    save_to_chroma(all_chunks, model_name, provider_name, chroma_path, logger)


if __name__ == "__main__":
    generate_data_store()
