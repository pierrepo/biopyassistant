"""Pydantic data models used to validate course configuration defined in YAML files."""

from pathlib import Path

from pydantic import BaseModel, Field, FilePath


class CourseChapter(BaseModel):
    """Validated definition of a course chapter."""

    id: str = Field(
        ...,
        description="Unique chapter identifier.",
    )
    title: str = Field(
        ...,
        description="Chapter title.",
    )
    source_file_path: FilePath = Field(
        ...,
        description="Path to the source markdown file for the chapter.",
    )
    processed_file_path: Path = Field(
        ...,
        description="Path to the processed markdown file for the chapter.",
    )


class CourseLevel(BaseModel):
    """Validated definition of a course level."""

    display_name: str = Field(
        ...,
        description="Human-readable name of the level for UI display.",
    )
    comment: str = Field(
        ...,
        description="Description of the level.",
    )
    chapters: list[str] = Field(
        ...,
        description="List of chapter identifiers associated with the level.",
        min_items=1,
    )
    prompt_path: FilePath = Field(
        ...,
        description="Path to the prompt template file for the level.",
    )
