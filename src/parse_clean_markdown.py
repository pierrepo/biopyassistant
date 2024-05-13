"""Module for processing Markdown files.

This module provides functions for processing Markdown files. It includes functionality to renumber headers and clean Python comments in Markdown content.

Usage:
======
    python src/parse_clean_markdown.py --in source_dir --out dest_dir

Where:
    source_dir : str
        The source directory containing Markdown files to be processed.
    dest_dir : str
        The destination directory to save the processed Markdown files.

Example:
========
    python src/parse_clean_markdown.py --in data/markdown_raw --out data/markdown_processed

This command will process Markdown files located in the 'data/markdown_raw' directory and save the processed files to the 'data/markdown_processed' directory.
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
import argparse
from typing import Union

from loguru import logger


# FUNCTIONS
def get_args() -> tuple[str, str]:
    """Get source and destination directories from command line arguments.

    Returns
    -------
    tuple[str, str]
        A tuple containing the source and destination directories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="source_dir",
        required=True,
        help="Source directory containing Markdown files.",
    )
    parser.add_argument(
        "--out",
        dest="dest_dir",
        required=True,
        help="Destination directory to save processed files.",
    )
    args = parser.parse_args()
    return args.source_dir, args.dest_dir


def clean_python_comments(content: str) -> str:
    """Remove spaces between '#' and comments in Python code blocks in Markdown content.

    Example:
    ```python
    # This is a comment.
    ```
    will be converted to:
    ```python
    #This is a comment.
    ```

    Parameters
    ----------
    content : str
        Content of Markdown file.

    Returns
    -------
    str
        Markdown content with spaces removed between '#' and comments in Python code blocks.
    """
    is_python_block = False
    cleaned_content = []
    modified_comment_lines = 0

    logger.info("Cleaning comments in Python code blocks...")

    for line in content.split("\n"):
        if line.strip().startswith("```python"):
            is_python_block = True
            cleaned_content.append(line)
        elif line.strip().startswith("```") and is_python_block:
            is_python_block = False
            cleaned_content.append(line)
        elif is_python_block:
            # Check if line has '#' followed by space(s) and remove them
            modified_line = re.sub(r"#\s+", "#", line)
            if modified_line != line:  # Check if any modification was made
                modified_comment_lines += 1
                cleaned_content.append(modified_line)
            else:
                cleaned_content.append(line)

        else:
            cleaned_content.append(line)

    logger.info(
        f"Number of space lines modified comment lines: {modified_comment_lines}"
    )
    logger.info(f"Number of final content lines: {len(cleaned_content)}")
    logger.success(f"Cleaning comments in Python code blocks complete.")

    return "\n".join(cleaned_content)


def renumber_headers(content: str, chapter_number: Union[int, str]) -> str:
    """Renumber headers in Markdown content.

    Parameters
    ----------
    content : str
        The Markdown content to renumber.
    chapter_number : Union[int, str]
        The chapter number to use for renumbering headers.

    Returns
    -------
    str
        The Markdown content with renumbered headers.
    """
    logger.info("Renumbering headers...")
    # Regex pattern to match headers with leading "#" and no following "#"
    header_pattern = r"^(#+)\s+([^#]*)$"
    # Define default header levels.
    # We should have no more than 4 levels of headers.
    headers = {
        1: chapter_number,  # Level 1: chapter / annexe.
        2: 0,  # Level 2: section.
        3: 0,  # Level 3: Sub-section
        4: 0,  # Level 4: Sub-sub-section.
    }
    # Stores the file content with renumbered headers
    processed_content = []
    for line in content.split("\n"):
        match = re.match(header_pattern, line)
        if match:
            header_level = len(match.group(1))
            header_text = match.group(2)
            # Show errors if we are above level 4
            if header_level > 4:
                logger.error("Header level beyond level 4!")
                logger.error(line)
                processed_content.append(line)
                continue
            # Increment the appropriate header level,
            # if below chapter / annexe level:
            if header_level != 1:
                headers[header_level] += 1
                # Reset subsequent levels
                for level in range(header_level + 1, len(headers) + 1):
                    headers[level] = 0
            # Create the header with numbers
            header_numbers = list(headers.values())[:header_level]
            header_numbers_as_str = ".".join([str(level) for level in header_numbers])
            line = f"{'#' * header_level} {header_numbers_as_str} {header_text}"
        processed_content.append(line)
    logger.success("Headers renumbered successfully.")
    return "\n".join(processed_content)


def process_md_files(source_dir: str, dest_dir: str) -> None:
    """Process Markdown files in the source directory and save them to the destination directory.

    Parameters
    ----------
    source_dir : str
        The source directory containing Markdown files.
    dest_dir : str
        The destination directory to save processed files.
    """
    logger.info("Processing Markdown files...\n")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Get a sorted list of Markdown files in the source directory
    markdown_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".md")])

    for filename in markdown_files:
        logger.info(f"Processing file: {filename}")

        # get the path of the source and destination files
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        # Read the content of the source file
        with open(source_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Clean Python comments
        content = clean_python_comments(content)

        # Renumber headers
        if filename.startswith("annexe"):
            content = renumber_headers(content, "A")
        if re.match(r"\d{2}_", filename):
            chapter_number = int(filename.split("_")[0])
            content = renumber_headers(content, chapter_number)

        # Save the processed content to the destination file
        with open(dest_path, "w", encoding="utf-8") as file:
            file.write(content)

    logger.success("Markdown files processed successfully.\n")


# MAIN PROGRAM
if __name__ == "__main__":
    source_dir, dest_dir = get_args()  # Get source and destination directories
    process_md_files(source_dir, dest_dir)
