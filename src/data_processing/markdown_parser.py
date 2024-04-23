"""Module for processing Markdown files.

This module provides functions for processing Markdown files. It includes functionality to renumber headers and clean Python comments in Markdown content.

Usage:
======
    python src/data_processing/markdown_parser.py [source_dir] [dest_dir]

Where:
    - source_dir : str, optional
        The source directory containing Markdown files to be processed. Default: RAW_DATA_PATH = "data/raw_python_courses"
    - dest_dir : str, optional
        The destination directory to save the processed Markdown files. Default: PROCESSED_PATH = "data/processed_python_courses"

Example:
========
    python src/data_processing/markdown_parser.py

This command will process Markdown files located in the 'data/raw_python_courses' directory and save the processed files to the 'data/processed_python_courses' directory.
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

from loguru import logger


# CONSTANTS
RAW_DATA_PATH = "data/raw_python_courses"
PROCESSED_PATH = "data/processed_python_courses"


# FUNCTIONS
def get_args() -> tuple[str, str]:
    """Get source and destination directories from command line arguments or defaults.

    Returns
    -------
    tuple[str, str]
        A tuple containing the source and destination directories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", nargs="?", default=RAW_DATA_PATH, help="The source directory containing Markdown files to be processed. Default: RAW_DATA_PATH")
    parser.add_argument("dest_dir", nargs="?", default=PROCESSED_PATH, help="The destination directory to save the processed Markdown files. Default: PROCESSED_PATH")
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

    logger.info(f"Number of space lines modified comment lines: {modified_comment_lines}")
    logger.info(f"Number of final content lines: {len(cleaned_content)}")
    logger.success(f"Cleaning comments in Python code blocks complete.")

    return "\n".join(cleaned_content)


def renumber_headers_courses(content: str, global_counters_courses: list[int]) -> str:
    """Renumber headers in Markdown content.

    Parameters
    ----------
    content : str
        The Markdown content to renumber.
    global_counters : list of int
        Global counters for each level of header.

    Returns
    -------
    str
        The Markdown content with renumbered headers.
    """
    logger.info("Renumbering headers for annexes...")
    header_pattern = r'^\s*(#+)\s+(?![#])\s*(.*)$'  # Regex pattern to match headers with leading '#' and no following '#'
    new_content = []  # Stores the new content with renumbered headers

    for line in content.split('\n'):
        match = re.match(header_pattern, line)
        if match:
            header_level = len(match.group(1))
            header_text = match.group(2)

            # Increment the appropriate global level and reset subsequent levels
            global_counters_courses[header_level - 1] += 1
            global_counters_courses[header_level:] = [0] * (4 - header_level)

            # Create the new header with renumbered level
            new_level = '.'.join(map(str, global_counters_courses[:header_level]))
            new_header = f"{'#' * header_level} {new_level} {header_text}"
            line = new_header

        new_content.append(line)

    logger.success("Headers renumbered.")

    return '\n'.join(new_content)


def renumber_headers_annexes(content: str, global_counters_annexes: list[int]) -> str:
    """Renumber headers in Markdown content for annexes.

    Parameters
    ----------
    content : str
        The Markdown content to renumber.
    global_counters_annexes : list of int
        Global counters for each level of header for annexes.

    Returns
    -------
    str
        The Markdown content with renumbered headers.
    """
    logger.info("Renumbering headers for annexes...")
    header_pattern = r'^\s*(#+)\s+(?![#])\s*(.*)$'  # Regex pattern to match headers with leading '#' and no following '#'
    new_content = []  # Stores the new content with renumbered headers

    for line in content.split('\n'):
        match = re.match(header_pattern, line)
        if match:
            header_level = len(match.group(1))
            header_text = match.group(2)

            # Get the appropriate letter for the header level
            letter = chr(ord('B') + global_counters_annexes[header_level - 1] - 1)

            # Increment the appropriate global level and reset subsequent levels
            global_counters_annexes[header_level - 1] += 1
            global_counters_annexes[header_level:] = [0] * (4 - header_level)

            # Create the new header with renumbered level
            new_level = f"{letter}" if header_level == 1 else f"A.{'.'.join(map(str, global_counters_annexes[1:header_level]))}"
            new_header = f"{'#' * header_level} {new_level} {header_text}"
            line = new_header

        new_content.append(line)

    logger.success("Headers renumbered for annexes.")

    return '\n'.join(new_content) 


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

    # Initialise global counters for header levels
    global_counters_courses = [0, 0, 0, 0]
    global_counters_annexes = [0, 0, 0, 0]

    # Get a sorted list of Markdown files in the source directory
    markdown_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.md')])

    for filename in markdown_files:
        logger.info(f"Processing file: {filename}")
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        with open(source_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Clean Python comments
        content = clean_python_comments(content)

        if filename.startswith("annexe"):
            content = renumber_headers_annexes(content, global_counters_annexes)
        else:
            # Renumber headers with global counters
            content = renumber_headers_courses(content, global_counters_courses)

        with open(dest_path, 'w', encoding='utf-8') as file:
            file.write(content)

    logger.success("Markdown files processed successfully.\n")


# MAIN PROGRAM
if __name__ == "__main__":
    source_dir, dest_dir = get_args() # Get source and destination directories if provided
    process_md_files(source_dir, dest_dir)
