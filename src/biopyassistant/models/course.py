"""Pydantic data models used to validate course configuration defined in YAML files."""

from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field, FilePath, field_validator


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
    prompt_file: str = Field(
        ...,
        description="Name of the prompt template file for the level.",
    )

    @field_validator(
        "prompt_file",
        mode="before",
    )
    @classmethod
    def check_file_exists(cls, v: str) -> str:
        """Ensure the prompt file exists in the packaged templates.

        Parameters
        ----------
        v : str
            The name of the prompt template file.

        Returns
        -------
        str
            The validated prompt file name.

        Raises
        ------
        ValueError
            If the specified prompt file does not exist in the package.
        """
        if not resources.files("biopyassistant.prompt_templates").joinpath(v).exists():
            msg = f"Prompt file '{v}' does not exist in biopyassistant.prompt_templates"
            raise ValueError(msg)
        return v
