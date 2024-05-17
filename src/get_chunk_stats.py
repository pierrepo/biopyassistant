"""Save details of chunks to a text file and/or save the number of tokens and chunks for each files to a CSV file.

Usage:
------
    python src/get_chunk_stats.py --data_dir [data_dir] --chroma_path [chroma_path] [--txt_output <txt_output>] [--csv_output <csv_output>]

Arguments:
--------
    --data_dir : str
        The path to the directory containing the Markdown documents.
    --chroma_path : str
        The path to the directory containing the Chroma database.
    --txt_output : str, optional
        The name of the output text file to save the chunks with metadatas.
    --csv_output : str, optional
        The name of the output file to save number of tokens and chunks for each Markdown files contained in the data directory. 

Example:
--------
    python src/get_chunk_stats.py --data_dir "data/markdown_processed" --chroma_path "chroma_db" --txt_output "chunks_details_no_overlap" --csv_output "nb_tokens_chunks_no_overlap"

This command will load the Markdown documents from the 'data/markdown_processed' directory, load the Chroma database from the 'chroma_db' directory, 
reconstruct the chunks from the vector database, save the details of each chunk to a text file named 'chunks_details_no_overlap.txt' 
and save the number of tokens and chunks for each Markdown files to a CSV file named 'nb_tokens_chunks_no_overlap.csv'.
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import sys
import argparse
from statistics import mean
from typing import List, Tuple

import tiktoken
from loguru import logger
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# MODULE IMPORTS
from create_database import load_documents, get_file_names
from query_chatbot import load_database


# FUNCTIONS
def get_args() -> Tuple[str, str, str, str]:
    """Get the command line arguments.

    Returns
    -------
    data_dir : str
        The path to the directory containing the Markdown documents.
    chroma_path : str
        The path to the directory containing the Chroma database.
    txt_output : str
        The name of the output text file to save the chunks with metadatas.
    csv_output : str
        The name of the output file to save number of tokens and chunks for each Markdown files contained in the data directory.
    """
    logger.info("Getting the command line arguments...")
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Save details of chunks to a text file and save the number of tokens and chunks for each files to a CSV file."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        help="The path to the directory containing the Markdown documents.",
    )
    parser.add_argument(
        "--chroma_path",
        type=str,
        help="The path to the directory containing the Chroma database.",
    )
    parser.add_argument(
        "--txt_output",
        type=str,
        default=None,
        help="The name of the output file to save the text chunks with metadata.",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        default=None,
        help="The name of the output file to save number of tokens and chunks for each Markdown files contained in the data directory.",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Check the required arguments
    if args.data_dir == None:
        logger.error(
            "Please specify the path to the directory containing the Markdown documents."
        )
        sys.exit(1)
    if args.chroma_path == None:
        logger.error(
            "Please specify the path to the directory containing the Chroma database."
        )
        sys.exit(1)

    # Check if the directories exist
    if not os.path.exists(args.data_dir):
        logger.error(
            f"The directory '{args.data_dir}' specified by --data_dir does not exist."
        )
        sys.exit(1)  # Exit the program
    if not os.path.exists(args.chroma_path):
        logger.error(
            f"The directory '{args.chroma_path}' specified by --chroma_path does not exist."
        )
        sys.exit(1)  # Exit the program

    # Warnings if none of the output files are specified
    if args.txt_output == None and args.csv_output == None:
        logger.error(
            "No output file specified. Please specify at least one output file."
        )
        sys.exit(1)  # Exit the program

    # Log information for output files
    if args.txt_output != None:
        logger.info(
            f"The details of each chunk will be saved in {args.txt_output}.txt."
        )
    if args.csv_output != None:
        logger.info(
            f"The number of tokens and chunks for each Markdown file will be saved in {args.csv_output}.csv."
        )

    logger.success("Got the command line arguments successfully.\n")

    return args.data_dir, args.chroma_path, args.txt_output, args.csv_output


def add_file_names_to_metadata(documents: List[Document]) -> List[Document]:
    """Add the file names in the metadata of the documents.

    Parameters
    ----------
    documents : list of Document
        List of documents to add the file names in the metadata.
        format : [{"page_content": str, "metadata": dict}, ...]

    Returns
    -------
    documents : list of Document
        List of documents with the file names in the metadata.
    """
    logger.info("Adding the file names in the metadata of the documents...")
    logger.info(f"{type(documents[0])}")
    # Add the file names in the metadata of the documents
    for document in documents:
        # Extract the file name from the metadata source
        source = document.metadata.get("source", "")
        if source:
            file_name = source.split("/")[-1].split(".")[
                0
            ]  # Extract the file name without extension
            document.metadata["file_name"] = file_name

    logger.success(
        "Added the file names in the metadata of the documents successfully.\n"
    )

    return documents


def add_nb_tokens_to_metadata(documents: List[Document]) -> List[Document]:
    """Add the number of tokens in the metadata of the documents.

    Parameters
    ----------
    documents : list of Document
        List of documents to add the number of tokens in the metadata.

    Returns
    -------
    documents : list of Document
        List of documents with the number of tokens in the metadata.
    """
    logger.info("Adding the number of tokens in the metadata of the documents...")

    # Get the encoding for tokenization
    # for openai embeddings
    encoding = tiktoken.get_encoding("cl100k_base")

    # Add the number of tokens in the metadata of the documents
    for document in documents:
        # Get the number of tokens in the document
        nb_tokens = len(encoding.encode(document.page_content))
        document.metadata["nb_tokens"] = nb_tokens

    logger.success(
        "Added the number of tokens in the metadata of the documents successfully.\n"
    )

    return documents


def reconstruct_chunks(vector_db: Chroma) -> List[Document]:
    """Reconstruct the chunks from the vector database.

    Parameters
    ----------
    vector_db : Chroma
        The vector database to reconstruct the chunks.

    Returns
    -------
    chunks : list of Document
        List of text chunks reconstructed from the vector database.
        format : [{"page_content": str, "metadata": dict}, ...]
    """
    logger.info("Reconstructing the chunks from the vector database...")

    chunks = []
    vector_collections = vector_db.get()
    total_chunks = len(vector_collections["ids"])
    for i in range(total_chunks):
        chunk = {
            "page_content": vector_collections["documents"][i],
            "metadata": vector_collections["metadatas"][i],
        }
        # ** used to unpack a dictionary into keyword arguments.
        chunk = Document(**chunk)
        chunks.append(chunk)

    logger.success(
        f"Reconstructed {total_chunks} chunks from the vector database successfully.\n"
    )

    return chunks


def save_to_txt(chunks: List[Document], txt_output: str) -> None:
    """Save text chunks to a text file with metadata.

    Parameters
    ----------
    chunks : list of Document
        List of text chunks to save to a text file.
    txt_output : str
        The name of the output file to save the text chunks with metadata.
    """
    logger.info(f"Saving into text file...")

    txt_output_path = txt_output + ".txt"  # add .txt extension

    # Get statistics of the tokens for all the chunks
    all_tokens = sum(chunk.metadata.get("nb_tokens", 0) for chunk in chunks)
    mean_tokens = mean(chunk.metadata.get("nb_tokens", 0) for chunk in chunks)
    min_tokens = min(chunk.metadata.get("nb_tokens", 0) for chunk in chunks)
    max_tokens = max(chunk.metadata.get("nb_tokens", 0) for chunk in chunks)

    # Save the details of the chunks to a text file
    with open(txt_output_path, "w") as f:
        f.write("Chunks Details :\n\n")
        # statistics of the tokens for all the chunks
        f.write("Statistics of the tokens for all the chunks:\n")
        f.write(f"- Count : {all_tokens}\n")
        f.write(f"- Mean : {round(mean_tokens, 3)}\n")
        f.write(f"- Min : {min_tokens}\n")
        f.write(f"- Max : {max_tokens}\n\n")

        # sort the chunks by the id
        chunks = sorted(chunks, key=lambda x: x.metadata["id"])

        for chunk in chunks:
            f.write(f"Chunk id: {chunk.metadata['id']}\n")
            f.write(f"Number of Tokens: {chunk.metadata['nb_tokens']}\n")
            f.write(f"Url: {chunk.metadata['url']}\n")
            f.write(f"File Name: {chunk.metadata['file_name']}\n")
            f.write(f"Chapter Name: {chunk.metadata['chapter_name']}\n")
            if "section_name" in chunk.metadata:
                f.write(f"Section Name: {chunk.metadata.get('section_name', '')}\n")
            if "subsection_name" in chunk.metadata:
                f.write(
                    f"Subsection Name: {chunk.metadata.get('subsection_name', '')}\n"
                )
            if "subsubsection_name" in chunk.metadata:
                f.write(
                    f"Subsubsection Name: {chunk.metadata.get('subsubsection_name', '')}\n"
                )
            f.write(f"Content:\n")
            f.write(f"{chunk.page_content}\n\n")

    logger.success(
        f"Saved the details of each chunks successfully to '{txt_output_path}'.\n"
    )


def save_to_csv(
    file_names: List[str],
    documents: List[Document],
    chunks: List[Document],
    csv_output: str,
) -> None:
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
        f.write("filename, token_number, chunk_number, token_number_from_chunks\n")
        for filename in file_names:
            # get the number of tokens in the metadata of the documents
            for document in documents:
                if document.metadata.get("file_name", "") == filename:
                    token_number = document.metadata.get("nb_tokens", 0)

            # get the number of chunks and tokens from the chunks
            token_number_from_chunks = 0
            chunk_number = 0
            for chunk in chunks:
                if chunk.metadata.get("file_name", "") == filename:
                    token_number_from_chunks += chunk.metadata.get("nb_tokens", 0)
                    chunk_number += 1
            f.write(
                f"{filename}, {token_number}, {chunk_number}, {token_number_from_chunks}\n"
            )

    logger.success(
        f"Saved the number of tokens and chunks for each files successfully to '{csv_output_path}'.\n"
    )


def main() -> None:
    """Main function to save details of chunks to a text file and save the number of tokens and chunks for each files to a CSV file."""
    # Get the command line arguments
    data_dir, chroma_path, txt_output, csv_output = get_args()

    # load documents from the specified directory
    documents = load_documents(data_dir)

    # get the file names of the documents
    file_names = get_file_names(documents)

    # add the file names in the metadata of the documents
    documents_with_file_names = add_file_names_to_metadata(documents)

    # get the number of tokens for each document
    documents_with_nb_tokens = add_nb_tokens_to_metadata(documents_with_file_names)

    # load the Chroma database
    vector_db = load_database(chroma_path)[0]

    # get the chunks from the vector database
    chunks = reconstruct_chunks(vector_db)

    # save the details of each chunks to a text file
    if txt_output != None:
        save_to_txt(chunks, txt_output)

    # save the number of tokens and chunks for each files to a CSV file
    if csv_output != None:
        save_to_csv(file_names, documents_with_nb_tokens, chunks, csv_output)


# MAIN PROGRAM
if __name__ == "__main__":
    main()
