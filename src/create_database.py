"""Creates the ChromaDB database from Markdown files in the specified directory.

This script loads Markdown files from the specified directory, concatenates their content, 
and splits the content into chunks based on headers and word limits. The resulting chunks are saved to a ChromaDB database.
It can also save the details of each chunks with metadatas to a text file. Additionally, it can save the number of tokens and chunks for each Markdown files to a CSV file.

Usage:
======
    python src/create_database.py --data_dir [data_dir] --chroma_out [chroma_output] --chunk_size [chunk_size] --chunk_overlap [chunk_overlap] --txt_out [txt_output] --csv_out [csv_output]

Options:
    data_dir : str, optional
        The directory containing the processed Markdown files of the python course. Default: PROCESSED_DATA_PATH"
    chroma_output : str, optional
        The name of the output path to save the ChromaDB database. Default: CHROMA_PATH.
    chunk_size : int, optional
        The size of the text chunks to be created. Default: 600.
    chunk_overlap : int, optional
        The overlap between text chunks. Default: 100.
    txt_output : str, optional
        The name of the output file to save the text chunks with metadata. Default: None.
    csv_output : str, optional
        The name of the output file to save the number of tokens for each files of the course. Default: None.

Example:
========
    python src/create_database.py --data_dir data/markdown_processed --chroma_out chroma_db --chunk_size 500 --chunk_overlap 50 --txt_out chunks_details --csv_out tokens_count

        
This command will create a ChromaDB database from the Markdown files in the 'data/markdown_processed' directory.
It will use a chunk size of 500 and an overlap of 50. The text chunks with metadata will be saved to 'chunks_details.txt', and the token count for each Markdown file will be saved to 'tokens_count.csv'.
The ChromaDB database will be saved to 'chroma_db'.
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
from statistics import mean

import tiktoken
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
def get_args() -> tuple[str,str,int,int,str,str]:
    """Parse command-line arguments.

    Returns
    -------
    data_dir, chroma_output_path, chunk_size, chunk_overlap, txt_output, csv_output : Tuple[str, str, int, int, str, str]
        - data_dir : str
            The directory containing the processed Markdown files of the python course.
        - chroma_output_path : str
            The name of the output path to save the ChromaDB database.
        - chunk_size : int
            The size of the text chunks to be created.
        - chunk_overlap : int
            The overlap between text chunks.
        - txt_output : str
            The name of the output file to save the details of chunks with metadata.
        - csv_output : str
            The name of the output file to save the number of tokens for each files of the course.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Create a ChromaDB database from Markdown files in the specified directory."
    )

    # Add the arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=PROCESSED_DATA_PATH,
        help="The directory containing the processed Markdown files of the python course.",
    )
    parser.add_argument(
        "--chroma_out",
        type=str,
        default=CHROMA_PATH,
        help="The name of the output path to save the ChromaDB database.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=600,
        help="The size of the text chunks to be created.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=100,
        help="The overlap between text chunks.",
    )
    parser.add_argument(
        "--txt_out",
        type=str,
        default=None,
        help="The name of the output file to save the text chunks with metadata.",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="The name of the output file to save the number of tokens for each files of the course.",
    )

    # Parse the arguments
    args = parser.parse_args()

    return (
        args.data_dir,
        args.chroma_out,
        args.chunk_size,
        args.chunk_overlap,
        args.txt_out,
        args.csv_out,
    )


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
    # print(chunks[100], end="\n\n")
    # print(chunks[3000], end="\n\n")

    return chunks


def adding_index_to_metadata(chunks: list[Document]) -> list[Document]:
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

    logger.success("Added index to metadata.\n")

    return chunks


def adding_tokens_to_metadata(chunks: list[Document]) -> list[Document]:
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
    encoding = tiktoken.get_encoding("cl100k_base") # for openai embeddings

    # Add the number of tokens to metadata of each chunk
    for chunk in chunks:
        # Encode the chunk content
        token = encoding.encode(chunk.page_content)
        # Count the number of tokens in the chunk content
        nb_tokens = len(token)
        chunk.metadata["nb_tokens"] = nb_tokens

    logger.success("Added the number of tokens to metadata successfully.\n")

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
        appendix_letter = re.search(r"\b[A-Z]+\s", chapter_name)
        # Corresponding chapter number or appendix letter with file name
        for file_name in file_names:
            if chapter_number and file_name.startswith(
                f"{chapter_number.group(0).strip().zfill(2)}_"
            ):  # zfill(2) to pad with zeros
                chunk.metadata["file_name"] = file_name
                break
            elif appendix_letter and file_name.endswith(f"_{appendix_letter.group()}"):
                chunk.metadata["file_name"] = file_name
                break

    logger.success("Added file names to metadata.\n")

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
    logger.info(f"Saving into text file...")

    txt_output_path = txt_output + ".txt"  # add .txt extension

    # Get statistics of the tokens for all the chunks
    all_tokens = sum(chunk.metadata.get('nb_tokens', 0) for chunk in chunks)
    mean_tokens = mean(chunk.metadata.get('nb_tokens', 0) for chunk in chunks)
    min_tokens = min(chunk.metadata.get('nb_tokens', 0) for chunk in chunks)
    max_tokens = max(chunk.metadata.get('nb_tokens', 0) for chunk in chunks)

    # Save the details of the chunks to a text file
    with open(txt_output_path, "w") as f:
        f.write(
            "Chunks were obtained with the following parameters, and here are the details:\n"
        )
        f.write(f"- Chunk size: {chunk_size}\n")
        f.write(f"- Chunk overlap: {chunk_overlap}\n\n")

        # statistics of the tokens for all the chunks
        f.write("Statistics of the tokens for all the chunks:\n")
        f.write(f"- Count : {all_tokens}\n")
        f.write(f"- Mean : {round(mean_tokens, 3)}\n")
        f.write(f"- Min : {min_tokens}\n")
        f.write(f"- Max : {max_tokens}\n\n")

        for chunk in chunks:
            f.write(f"Chunk id: {chunk.metadata.get('id', '')}\n")
            f.write(f"Number of Tokens: {chunk.metadata.get('nb_tokens', '')}\n")
            f.write(f"Url: {chunk.metadata.get('url', '')}\n")
            f.write(f"File Name: {chunk.metadata.get('file_name', '')}\n")
            f.write(f"Chapter Name: {chunk.metadata.get('chapter_name', '')}\n")
            f.write(f"Section Name: {chunk.metadata.get('section_name', '')}\n")
            f.write(f"Subsection Name: {chunk.metadata.get('subsection_name', '')}\n")
            f.write(
                f"Subsubsection Name: {chunk.metadata.get('subsubsection_name', '')}\n"
            )
            f.write(f"Content:\n")
            f.write(f"{chunk.page_content}\n\n")

    logger.success(f"Saved the details of each chunks successfully to '{txt_output_path}'.\n")


def save_to_csv(file_names: list[str], chunks: list[Document], csv_output: str) -> None:
    """Save the number of tokens and chunks for each files 

    Parameters
    ----------
    file_names : list of str
        List of file names of the Markdown documents.
    chunks : list of Document
        List of text chunks to save to a CSV file.
    csv_output : str
        The name of the output file to save the text chunks with metadata.
    """
    logger.info(f"Saving into CSV file...")

    csv_output_path = csv_output + ".csv"  # add .csv extension

    # Save the number of tokens and chunks for each files to a CSV file
    with open(csv_output_path, "w") as f:
        f.write("File name, Number of Tokens, Number of Chunks\n")
        for file_name in file_names:
            nb_tokens = 0
            nb_chunks = 0
            for chunk in chunks:
                if chunk.metadata.get("file_name", "") == file_name:
                    nb_tokens += chunk.metadata.get("nb_tokens", 0)
                    nb_chunks += 1
            f.write(f"{file_name}, {nb_tokens}, {nb_chunks}\n")

    logger.success(f"Saved the number of tokens and chunks for each files successfully to '{csv_output_path}'.\n")


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

    # Create a new DB from the documents.
    model_embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma.from_documents(
        chunks,
        model_embedding,
        persist_directory=chroma_output_path,
        collection_metadata={"hnsw:space": "cosine"},
    )  # distance metric
    db.persist()  # save the database to disk

    logger.success(f"Saved {len(chunks)} chunks to {chroma_output_path}.")


def generate_data_store() -> None:
    """Generates data store by loading, splitting text into chunks, adding metadata and saving the chunks to ChromaDB.
    it can also save the details of each chunks with metadatas to a text file and save the number of tokens for each files to a CSV file.
    """
    # get command-line arguments
    data_dir, chroma_output_path, chunk_size, chunk_overlap, txt_output, csv_output = get_args()

    # load documents from the specified directory and extract file names
    documents, file_names = load_documents(data_dir)

    # split text into chunks
    chunks = split_text(documents, chunk_size, chunk_overlap)

    # add index to the metadata
    chunks_with_index = adding_index_to_metadata(chunks)

    # add number of tokens to the metadata
    chunks_with_tokens = adding_tokens_to_metadata(chunks_with_index)

    # add file names to the chunks
    chunks_with_file_names = adding_file_names_to_metadata(chunks_with_tokens, file_names)

    # add URL to the chunks
    chunks_with_url = adding_url_to_metadata(chunks_with_file_names)

    # save the details of the chunks to a text file
    if txt_output is not None:
        save_to_txt(chunks_with_url, txt_output, chunk_size, chunk_overlap)

    # save the number of tokens for each files to a CSV file
    if csv_output is not None:
        save_to_csv(file_names,chunks_with_url, csv_output)

    # save the chunks to ChromaDB
    save_to_chroma(chunks_with_url, chroma_output_path)


# MAIN PROGRAM
if __name__ == "__main__":
    generate_data_store()
