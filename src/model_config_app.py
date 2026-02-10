"""Pydantic models used to validate the CLI argument of BioPyAssistant App."""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, DirectoryPath, Field, FilePath, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class CourseChapter(BaseModel):
    """Validated definition of a course chapter."""

    id: str = Field(
        ...,
        description="Unique chapter identifier.",
        min_length=1,
    )
    title: str = Field(
        ...,
        description="Human-readable chapter title.",
    )
    source_file_path: Path = Field(
        ...,
        description="Path to the raw Markdown source file.",
    )
    processed_file_path: Path = Field(
        ...,
        description="Path to the processed Markdown file.",
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


class LLMConfig(BaseModel):
    """Configuration model for Large Language Model (LLM) parameters."""

    llm_model_name: str = Field(
        ..., description="Default LLM model name to use if none is provided."
    )
    provider_llm_name: Literal["openai", "openrouter"] = Field(
        ..., description="Name of the LLM provider, must be 'openai' or 'openrouter'."
    )
    embeddings_model_name: str = Field(
        ...,
        description="Name of the embedding model to use for vector representations.",
    )
    vector_database_path: DirectoryPath = Field(
        ..., description="Path to the directory where the vector database is stored."
    )
    provider_embeddings_name: Literal["openai", "openrouter"] = Field(
        ...,
        description="Name of the embeddings model provider,"
        "must be 'openai' or 'openrouter'.",
    )
    prompt_path: FilePath = Field(
        ...,
        description="Path to the prompt template file used for generating responses.",
    )
    api_key_openai: SecretStr | None = Field(
        None,
        description="Secret API key for authenticating on openai provider.",
        env="OPENAI_API_KEY",
    )
    api_key_openrouter: SecretStr | None = Field(
        None,
        description="Secret API key for authenticating on openrouter provider.",
        env="OPENROUTER_API_KEY",
    )

    @property
    def api_key(self) -> SecretStr:
        """
        Return the correct API key based on the selected LLM provider.

        The returned key depends on the value of `self.provider_llm_name`:
        - If `provider_llm_name` is "openai", returns `api_key_openai`.
        - If `provider_llm_name` is "openrouter", returns `api_key_openrouter`.

        Returns
        -------
        SecretStr
            The API key corresponding to the currently selected provider.
        """
        if self.provider_llm_name == "openai":
            # if self.api_key_openai is None:
            #     msg = "OPENAI_API_KEY not set"
            #     raise ValueError(msg)
            return self.api_key_openai or SecretStr(os.getenv("OPENAI_API_KEY"))
        elif self.provider_llm_name == "openrouter":
            # print(self.api_key_openrouter)
            # print(os.getenv("OPENROUTER_API_KEY"))
            # if self.api_key_openrouter is None:
            #     msg = "OPENROUTER_API_KEY not set"
            #     raise ValueError(msg)
            return self.api_key_openrouter or SecretStr(os.getenv("OPENROUTER_API_KEY"))


class Settings(BaseSettings, cli_parse_args=True):
    """Global BioPyAssistant streamlit application settings."""

    # Basic application info
    app_name: str = Field(
        "BioPyAssistant", description="Application name displayed in the UI or logs."
    )
    app_description: str | None = Field(
        None, description="Short app description displayed in the logs."
    )
    app_version: str = Field("2.0", description="Current version of the application.")
    # Styling and assets
    css_path: FilePath = Field(
        ..., description="Path to the CSS file for Streamlit styling."
    )
    # Course configuration
    course_yaml_path: Path = Field(
        ...,
        description="Path to the YAML file defining course levels and chapters.",
    )
    # Nested configuration for LLM-related parameters
    llm: LLMConfig = Field(
        ..., description="Configuration for Large Language Model parameters."
    )

    # Compute the log path
    @property
    def log_path(self) -> Path:
        """Return the full path for today's log file."""
        log_file = (
            Path("logs")
            / f"{datetime.now().strftime('%Y%m%d')}"
            / "biopyassistant_app.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return log_file

    # Compute chapter and level information from the course YAML file
    @property
    def course_chapters(self) -> dict[str, CourseChapter]:
        """Parse and validate course chapters from the YAML configuration.

        This property reads the YAML file specified by `course_yaml_path`, extracts
        the course chapters defined under the "chapters" section, and validates them
        against the `CourseChapter` model. It returns a dictionary mapping chapter IDs
        to their corresponding `CourseChapter` instances.

        Returns
        -------
        dict[str, CourseChapter]
            A dictionary where keys are chapter IDs
            and values are `CourseChapter` objects.

        Raises
        ------
        ValueError
            If the YAML file is missing the "chapters" section, if any chapter is missing
            required fields defined in the `CourseChapter` model, or if there are
            duplicate chapter IDs.
        """
        with self.course_yaml_path.open(encoding="utf-8") as f:
            course_data = yaml.safe_load(f) or {}

        raw_chapters = course_data.get("chapters")
        if not raw_chapters:
            msg = "No 'chapters' section found in course YAML file."
            raise ValueError(msg)

        chapters: dict[str, CourseChapter] = {}

        for chapter in raw_chapters:
            try:
                chapter_model = CourseChapter(
                    id=str(chapter["id"]),
                    title=chapter["title"],
                    source_file_path=Path(chapter["source_file_path"]),
                    processed_file_path=Path(chapter["processed_file_path"]),
                )
            except KeyError as exc:
                msg = f"Missing required field {exc!s} in chapter definition."
                raise ValueError(msg) from exc

            if chapter_model.id in chapters:
                msg = f"Duplicate chapter id '{chapter_model.id}' detected."
                raise ValueError(msg)

            chapters[chapter_model.id] = chapter_model

        return chapters

    @property
    def course_levels(self) -> dict[str, CourseLevel]:
        """Parse and validate course levels from the YAML configuration.

        This property reads the YAML file specified by `course_yaml_path`, extracts
        the course levels defined under the "levels" section, and validates them
        against the `CourseLevel` model. It returns a dictionary mapping level names
        to their corresponding `CourseLevel` instances.

        Returns
        -------
        dict[str, CourseLevel]
            A dictionary where keys are level names
            and values are `CourseLevel` objects.

        Raises
        ------
        ValueError
            If the YAML file is missing the "levels" section, if any level is missing
            the required "name" field, or if any level is missing required fields
            defined in the `CourseLevel` model.
        """
        with self.course_yaml_path.open(encoding="utf-8") as f:
            course_data = yaml.safe_load(f) or {}

        raw_levels = course_data.get("levels")
        if not raw_levels:
            msg = "No 'levels' section found in course YAML file."
            raise ValueError(msg)

        levels: dict[str, CourseLevel] = {}

        for level in raw_levels:
            level_name = level.get("name")
            if not level_name:
                msg = "Each level must define a non-empty 'name' field."
                raise ValueError(msg)

            try:
                levels[level_name] = CourseLevel(
                    display_name=level["display_name"],
                    comment=level["comment"],
                    chapters=[str(ch) for ch in level["chapters"]],
                )
            except KeyError as exc:
                msg = f"Missing required field {exc!s} in level '{level_name}'."
                raise ValueError(msg) from exc

        return levels

    @property
    def validated_course_levels(self) -> dict[str, CourseLevel]:
        """Return course levels after validating chapter references.

        Returns
        -------
        dict[str, CourseLevel]
            A dictionary where keys are level names and values are `CourseLevel` objects
            that have been validated to ensure all referenced chapter IDs exist in the
            course chapters.

        Raises
        ------
        ValueError
            If any level references chapter IDs that are not defined in the
            course chapters.
        """
        chapters = self.course_chapters
        levels = self.course_levels

        for level_name, level in levels.items():
            missing_chapters = sorted(
                chapter_id
                for chapter_id in level.chapters
                if chapter_id not in chapters
            )

            if missing_chapters:
                msg = (
                    f"Level '{level_name}' references unknown chapter IDs: "
                    f"{missing_chapters}"
                )
                raise ValueError(msg)

        return levels

    # Load settings from this TOML file
    model_config = SettingsConfigDict(toml_file=["config_app.toml"])

    # Reorder the priority of different settings sources
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize the order and type of settings sources.

        Parameters
        ----------
        settings_cls : type[BaseSettings]
            The settings class being instantiated. Used by sources that
            need access to the settings schema.
        init_settings : PydanticBaseSettingsSource
            Source providing values passed directly to the Settings
            constructor.
        env_settings : PydanticBaseSettingsSource
            Source providing values from environment variables.

        Returns
        -------
        tuple[PydanticBaseSettingsSource, ...]
            Ordered tuple of settings sources. Earlier sources have higher
            priority during value resolution.
        """
        return (
            # Load settings from TOML files first
            TomlConfigSettingsSource(settings_cls),
            # Override with explicit constructor arguments
            init_settings,
            # Finally allow environment variables to override everything
            env_settings,
        )
