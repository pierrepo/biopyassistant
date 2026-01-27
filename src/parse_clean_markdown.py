"""Module for processing Markdown files.

This module provides functions for processing Markdown files.
It includes functionality to renumber headers and clean Python comments
embedded in Markdown content.

For example:

Header renumbering:
-------------------
Input:
    # Introduction
    ## Qu'est-ce que Python ?

Output:
    # 1 Introduction
    ## 1.1 Qu'est-ce que Python ?

Python comment cleaning:
------------------------
Input:
    ```python
    # Votre premier commentaire en  Python.
    print("Hello world!")

    # D'autres commandes plus utiles pourraient suivre.
    ```

Output:
    ```python
    # Votre premier commentaire en  Python.
    print("Hello world!")

    # D'autres commandes plus utiles pourraient suivre.
    ```

Usage:
======
    python src/parse_clean_markdown.py --in source_dir --out dest_dir

Where:
    source_dir : Path
        The source directory containing Markdown files to be processed.
    dest_dir : Path
        The destination directory to save the processed Markdown files.

Example:
========
    python src/parse_clean_markdown.py --in data/markdown_raw \
                                        --out data/markdown_processed

This command processes Markdown files located in the 'data/markdown_raw'
directory and saves the cleaned and renumbered files to the
'data/markdown_processed' directory.
"""

import re
from pathlib import Path

import click
from loguru import logger


def clean_python_comments(content: str) -> str:
    """Remove spaces between '#' and comments in Python code blocks in Markdown content.

    Example:
    ```python
    # This is a comment.
    ```
    will be converted to:
    ```python
    # This is a comment.
    ```

    Parameters
    ----------
    content : str
        Content of Markdown file.

    Returns
    -------
    str
        Markdown content with spaces removed between '#'
        and comments in Python code blocks.
    """
    is_python_block = False
    cleaned_content = []
    modified_comment_lines = 0

    logger.info("Cleaning comments in Python code blocks...")

    for line in content.split("\n"):
        # Check for the start or end of a Python code block
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
    logger.success("Cleaning comments in Python code blocks complete.")

    return "\n".join(cleaned_content)


def renumber_headers(content: str, header_prefix: int | str) -> str:
    """Renumber headers in Markdown content.

    Parameters
    ----------
    content : str
        The Markdown content to renumber.
    header_prefix : Union[int, str]
        The chapter number or appendix letter to use for renumbering headers.

    Returns
    -------
    str
        The Markdown content with renumbered headers.
    """
    logger.info("Renumbering headers...")
    # Regex pattern to match headers with leading "#" and no following "#"
    header_pattern = r"^(#+)\s+([^#]*)$"
    # Define default header levels.
    # We should have no more than 5 levels of headers.
    headers = {
        1: header_prefix,  # Level 1: chapter / appendix.
        2: 0,  # Level 2: section.
        3: 0,  # Level 3: Sub-section
        4: 0,  # Level 4: Sub-sub-section.
        5: 0,  # Level 5: Sub-sub-sub-section.
    }
    # Stores the file content with renumbered headers
    processed_content = []
    for line in content.split("\n"):
        match = re.match(header_pattern, line)
        if match:
            header_level = len(match.group(1))
            header_text = match.group(2)
            # Show errors if we are above level 5
            if header_level > 5:
                logger.error("Header level beyond level 5!")
                logger.error(line)
                processed_content.append(line)
                continue
            # Increment the appropriate header level,
            # if below chapter / appendix level:
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


@click.command()
@click.option(
    "--in",
    "source_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Source directory containing Markdown files.",
)
@click.option(
    "--out",
    "dest_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Destination directory to save processed files.",
)
def process_md_files(source_dir: Path, dest_dir: Path) -> None:
    """Process Markdown files in the source directory and save them to the destination directory.

    Parameters
    ----------
    source_dir : Path
        The source directory containing Markdown files.
    dest_dir : Path
        The destination directory to save processed files.
    """
    logger.info("Processing Markdown files...")

    # Create the output directory if it does not exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all Markdown files in sorted order
    for source_path in sorted(source_dir.glob("*.md")):
        logger.info(f"Processing file: {source_path.name}")

        # Read file content
        content = source_path.read_text(encoding="utf-8")

        # Remove or normalize Python-style comments
        content = clean_python_comments(content)

        # Handle appendix files (e.g., annexe_A_*.md)
        if source_path.name.startswith("annexe"):
            annex_character = source_path.stem.split("_")[1]
            content = renumber_headers(content, annex_character)

        # Handle numbered chapter files (e.g., 01_intro.md)
        elif re.match(r"\d{2}_", source_path.name):
            chapter_number = int(source_path.stem.split("_")[0])
            content = renumber_headers(content, chapter_number)

        # Write processed content to destination directory
        dest_path = dest_dir / source_path.name
        dest_path.write_text(content, encoding="utf-8")

    logger.success("Markdown files processed successfully")


# MAIN PROGRAM
if __name__ == "__main__":
    process_md_files()
