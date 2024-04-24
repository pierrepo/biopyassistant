"""Creates the ChromaDB database from Markdown files in the specified directory.

This script loads Markdown files from the specified directory, concatenates their content, 
and splits the content into chunks based on headers and word limits. The resulting chunks are saved to a ChromaDB database.

Usage:
======
    python src/create_database.py [data_dir] [chunk_size] [chunk_overlap] [txt_output]

Options:
    data_dir : str, optional
        The directory containing the Markdown files to be processed. Default: PROCESSED_DATA_PATH"
    chunk_size : int, optional
        The size of the text chunks to be created. Default: 300.
    chunk_overlap : int, optional
        The overlap between text chunks. Default: 100.
    txt_output : str, optional
        The name of the output file to save the text chunks with metadata. Default: None.
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
import argparse
import unicodedata

from loguru import logger
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# CONSTANTS
CHROMA_PATH = "chroma_db"
PROCESSED_DATA_PATH = "data/markdown_processed"


# FUNCTIONS
def get_args() -> tuple[str, int, int]:
    """Parse command-line arguments.

    Returns
    -------
    data_dir : str
        The directory containing the Markdown files to be processed.
    chunk_size : int
        The size of the text chunks to be created.
    chunk_overlap : int
        The overlap between text chunks.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=PROCESSED_DATA_PATH,
        help="The directory containing the Markdown files to be processed. Default: PROCESSED_DATA_PATH",
    )
    parser.add_argument(
        "chunk_size",
        nargs="?",
        default=300,
        type=int,
        help="The size of the text chunks to be created. Default: 300.",
    )
    parser.add_argument(
        "chunk_overlap",
        nargs="?",
        default=100,
        type=int,
        help="The overlap between text chunks. Default: 100.",
    )
    parser.add_argument(
        "txt_output",
        nargs="?",
        default=None,
        help="The output file to save the text chunks with metadata. Default: None.",
    )
    args = parser.parse_args()

    return args.data_dir, args.chunk_size, args.chunk_overlap, args.txt_output


def load_documents(data_dir: str) -> tuple[str, list[str]]:
    """Load Markdown documents, concatenate their content, and extract the name of the Markdown files.

    Parameters
    ----------
    data_dir : str
        The directory containing the Markdown files to be processed.

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
        data_dir, glob="*.md", show_progress=True, loader_cls=TextLoader
    )
    documents = loader.load()

    for document in documents:
        # Add the document content to the concatenated content
        concatenated_content += document.page_content + "\n"

        # Extract the file name from the metadata source
        source = document.metadata.get("source", "")
        if source:
            file_name = source.split("/")[-1].split(".")[
                0
            ]  # Extract the file name without extension
            file_names.append(file_name)

    logger.success(f"Markdown document loading complete.\n")

    return concatenated_content, file_names


def split_text(
    content: str, chunk_size: int, chunk_overlap: int
) -> list[Document]:
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
        # split on paragraphs, sentences, and words
        separators=["\n\n", "\n", " "],
    )

    # Split the resulting chunks further based on character limits
    chunks = text_splitter.split_documents(md_header_splits)

    logger.success(f"Split documents into {len(chunks)} chunks.\n")

    # tests
    # print(chunks[1], end="\n\n")
    # print(chunks[60], end="\n\n")
    # print(chunks[100], end="\n\n")
    # print(chunks[3000], end="\n\n")

    return chunks


def adding_file_names_to_metadata(
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
        appendix_letter = re.search(r"^\d+[A-Z]$", chapter_name)
        # Corresponding chapter number or appendix letter with file name
        for file_name in file_names:
            if chapter_number and file_name.startswith(
                f"{chapter_number.group(0).strip().zfill(2)}_"
            ):  # zfill(2) to pad with zeros
                chunk.metadata["file_name"] = file_name
                break
            elif appendix_letter and file_name.endswith(f"_{appendix_letter.group(0)}"):
                chunk.metadata["file_name"] = file_name
                break

    logger.success("Added file names to metadata.\n")

    return chunks


def preprocess_for_url(text: str) -> str:
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

    # Remove articles like 'de', 'le', etc.
    # processed_text = re.sub(r'\b(de|le|la|les|des|du|au|aux)\b', '', processed_text)

    # Remove points
    processed_text = processed_text.replace(".", "")

    # Replace spaces with hyphens
    processed_text = processed_text.replace(" ", "-")

    # Add a '#' at the beginning
    processed_text = "#" + processed_text

    return processed_text


def adding_url_to_metadata(chunks: list[Document]) -> list[Document]:
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
                chunk.metadata.get("subsubsection_name", "")
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

    logger.success("Added URL to metadata.\n")

    return chunks


def save_to_txt(
    chunks: list[Document], txt_output: str, chunk_size: int, chunk_overlap: int
) -> None:
    """Save text chunks to a text file with metadata.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to save to a text file.
    txt_output : str
        The name of the output file to save the text chunks with metadata.
    chunk_size : int
        The size of the text chunks to be created.
    chunk_overlap : int
        The overlap between text chunks.
    """
    logger.info(f"Saving text chunks to {txt_output}...")

    txt_output_path = txt_output + ".txt"  # add .txt extension

    # Save the details of the chunks to a text file
    with open(txt_output_path, "w") as f:
        f.write(
            "Chunks were obtained with the following parameters, and here are the details:\n"
        )
        f.write(f"- Chunk size: {chunk_size}\n")
        f.write(f"- Chunk overlap: {chunk_overlap}\n\n")
        for index, chunk in enumerate(chunks):
            f.write(f"Chunk {index + 1}:\n")
            f.write(f"Url: {chunk.metadata.get('url', '')}\n")
            f.write(f"File Name: {chunk.metadata.get('file_name', '')}\n")
            f.write(f"Chapter Name: {chunk.metadata.get('chapter_name', '')}\n")
            f.write(f"Section Name: {chunk.metadata.get('section_name', '')}\n")
            f.write(f"Subsection Name: {chunk.metadata.get('subsection_name', '')}\n")
            f.write(
                f"Subsubsection Name: {chunk.metadata.get('subsubsection_name', '')}\n"
            )
            f.write(f"Content: {chunk.page_content}\n\n")

    logger.success(f"Saved text chunks to {txt_output_path}.")


def save_to_chroma(chunks: list[Document]) -> None:
    """Save text chunks to ChromaDB.

    Parameters
    ----------
    chunks : list of str
        List of text chunks to save to ChromaDB.
    """
    logger.info("Saving to Chroma...")

    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    model_embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma.from_documents(chunks, model_embedding, persist_directory=CHROMA_PATH)
    db.persist()  # save the database to disk

    logger.success(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_data_store() -> None:
    """Generates data store by loading, splitting text into chunks, and saving the chunks to ChromaDB."""
    # get command-line arguments
    data_dir, chunk_size, chunk_overlap, txt_output = get_args()

    # load documents from the specified directory and extract file names
    documents, file_names = load_documents(data_dir)

    # split text into chunks
    chunks = split_text(documents, chunk_size, chunk_overlap)

    # add file names to the chunks
    chunks_with_file_names = adding_file_names_to_metadata(chunks, file_names)

    # add URL to the chunks
    chunks_with_url = adding_url_to_metadata(chunks_with_file_names)

    # save the details of the chunks to a text file
    if txt_output is not None:
        save_to_txt(chunks_with_url, txt_output, chunk_size, chunk_overlap)

    # save the chunks to ChromaDB
    save_to_chroma(chunks_with_url)


# MAIN PROGRAM
if __name__ == "__main__":
    generate_data_store()
