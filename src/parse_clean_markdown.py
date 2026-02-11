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
    #Votre premier commentaire en  Python.
    print("Hello world!")

    #D'autres commandes plus utiles pourraient suivre.
    ```

Usage:
======
    uv run src/parse_clean_markdown.py --config path/to/chapters_and_levels.yaml

Where:
    config : Path
        Path to the YAML file defining all chapters and levels.
        This YAML file should include the chapter names, titles, and the paths to the
        source Markdown files and the destination paths for the processed files.

Example:
========
    uv run src/parse_clean_markdown.py --config data/chapters_and_levels.yaml

This command processes Markdown files listed in the YAML file and saves
the cleaned and renumbered files to the paths specified in the YAML file.
"""

import re
from datetime import datetime
from pathlib import Path

import click
import loguru
import yaml

from logger import create_logger


def build_chapter_paths(
    file_name: str,
    raw_base_path: Path,
    processed_base_path: Path,
) -> tuple[Path, Path]:
    """Build source and processed file paths for a chapter.

    Parameters
    ----------
    file_name : str
        The name of the Markdown file for the chapter (e.g., "01_introduction.md").
    raw_base_path : Path
        The base path for raw Markdown files.
    processed_base_path : Path
        The base path for processed Markdown files.

    Returns
    -------
    tuple[Path, Path]
        A tuple containing the source path and the processed path for the chapter.
    """
    source_path = raw_base_path / file_name
    processed_path = processed_base_path / file_name
    return source_path, processed_path


def load_chapters_from_yaml(
    yaml_path: Path, logger: "loguru.Logger" = loguru.logger
) -> list[dict]:
    """Load chapters from a YAML file and construct paths.

    Parameters
    ----------
    yaml_path : Path
        Path to the YAML file defining chapters and levels.
    logger: "loguru.Logger"
        Logger for logging messages.

    Returns
    -------
    list[dict]
        A list of dictionaries with chapter information, including constructed paths
        for source and processed Markdown course files.
    """
    logger.info(f"Loading chapters and paths from YAML file: {yaml_path}...")
    try:
        with yaml_path.open("r", encoding="utf-8") as file:
            # Load YAML data
            data = yaml.safe_load(file) or {}
            # Get chapter information
            chapters = data.get("chapters", [])
            # Get base paths of raw and processed courses
            raw_base_path = Path(data.get("raw_course_base_path", ""))
            processed_base_path = Path(data.get("processed_course_base_path", ""))
            # Build source and processed paths for each chapter
            for chapter in chapters:
                file_name = chapter.get("file_name")
                if file_name:
                    chapter["source_path"], chapter["processed_path"] = (
                        build_chapter_paths(
                            file_name, raw_base_path, processed_base_path
                        )
                    )
                else:
                    logger.warning(
                        f"No file name defined for chapter: {chapter.get('id')}"
                    )

            logger.success(f"Loaded {len(chapters)} chapters successfully.")
            return chapters
    except FileNotFoundError:
        logger.error(f"YAML file not found: {yaml_path}")
        return []
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return []


def clean_python_comments(content: str, logger: "loguru.Logger" = loguru.logger) -> str:
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
    logger: "loguru.Logger"
        Logger for logging messages.

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
    logger.success("Python code block comments cleaned.")

    return "\n".join(cleaned_content)


def renumber_headers(
    content: str, header_prefix: int | str, logger: "loguru.Logger" = loguru.logger
) -> str:
    """Renumber headers in Markdown content.

    Parameters
    ----------
    content : str
        The Markdown content to renumber.
    header_prefix : Union[int, str]
        The chapter number or appendix letter to use for renumbering headers.
    logger: "loguru.Logger"
        Logger for logging messages.

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
    "--config",
    "yaml_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the YAML file defining chapters and levels."
    " This YAML file should include the chapter names, titles, and the paths to the "
    "source Markdown files and the destination paths for the processed files.",
)
def process_md_files(yaml_path: Path) -> None:
    """Process Markdown files listed in a YAML file and save them.

    Parameters
    ----------
    yaml_path : Path
        Path to the YAML file defining chapters and their Markdown filenames.
    """
    # Set up logging
    log_path = f"logs/{datetime.now().strftime('%Y%m%d')}/parse_clean_markdown.log"
    logger = create_logger(log_path)
    logger.info("Starting Markdown processing...")

    # Load chapters from YAML file
    chapters = load_chapters_from_yaml(yaml_path, logger)

    # Process each chapter's Markdown file
    saved_count = 0
    for i, chapter in enumerate(chapters, start=1):
        chapter_name = f"{chapter.get('id')}. {chapter.get('title')}"
        logger.info(f"Chapter: {chapter_name} ({i}/{len(chapters)})")

        # Get source file path from YAML
        source_path = chapter.get("source_path")
        if not source_path:
            logger.warning(f"No source file defined for chapter {chapter_name}")
            continue
        # Check if source file exists
        if not source_path.exists():
            logger.warning(f"Markdown file not found: {source_path}")
            continue
        logger.info(f"Processing file: {source_path.name}")
        # Read content
        content = source_path.read_text(encoding="utf-8")
        # Clean Python-style comments
        content = clean_python_comments(content, logger)
        # Renumber headers
        # Determine if it's an annex or a chapter based on filename
        stem = source_path.stem.lower()
        # Annex files are named with "annexe" followed by an underscore + letter
        # For example: "annexe_a.md", "annexe_b.md", etc.
        if stem.startswith("annexe"):
            annex_character = stem.split("_")[1]
            content = renumber_headers(content, annex_character, logger)
        # Chapter files are named with a number followed by an underscore
        # For example: "01_introduction.md", "02_variables.md", etc.
        elif re.match(r"\d{2}_", source_path.name):
            chapter_number = int(stem.split("_")[0])
            content = renumber_headers(content, chapter_number, logger)

        # Get destination file path from YAML
        dest_path = chapter.get("processed_path")
        if not dest_path:
            logger.warning(f"No processed file path defined for chapter {chapter_name}")
            continue
        # Create output directory if it does not exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Save processed content to destination path
        try:
            dest_path.write_text(content, encoding="utf-8")
            logger.info(f"Processed file saved to: {dest_path}")
            saved_count += 1
        except PermissionError:
            logger.error(f"Permission denied when writing file: {dest_path}")
        except OSError as e:
            logger.error(f"OS error when writing file {dest_path}: {e}")

    logger.success(
        f"Saved {saved_count}/{len(chapters)} processed Markdown files successfully!"
    )


# MAIN PROGRAM
if __name__ == "__main__":
    process_md_files()
