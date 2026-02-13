"""Pydantic data models used to validate UI Streamlit application settings."""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, DirectoryPath, Field, FilePath, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from models.course import CourseChapter, CourseLevel
from parse_clean_markdown import load_chapters_from_yaml
from query_chatbot import get_level_infos


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

        Raises
        ------
        ValueError
            If the selected provider is "openai" but `api_key_openai` is not
            set, or if the selected provider is "openrouter" but `api_key_openrouter`
            is not set.
        """
        if self.provider_llm_name == "openai":
            if self.api_key_openai is None:
                msg = "OPENAI_API_KEY not set"
                raise ValueError(msg)
            return self.api_key_openai or SecretStr(os.getenv("OPENAI_API_KEY"))
        elif self.provider_llm_name == "openrouter":
            if self.api_key_openrouter is None:
                msg = "OPENROUTER_API_KEY not set"
                raise ValueError(msg)
            return self.api_key_openrouter or SecretStr(os.getenv("OPENROUTER_API_KEY"))


class Settings(BaseSettings, cli_parse_args=True):
    """Global BioPyAssistant streamlit application settings."""

    # Basic application info
    app_name: str = Field(
        ..., description="Application name displayed in the UI or logs."
    )
    app_description: str | None = Field(
        None, description="Short app description displayed in the logs."
    )
    app_version: str = Field(..., description="Current version of the application.")
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
        """
        chapters = load_chapters_from_yaml(self.course_yaml_path, ui_logger=True)
        # Convert list of CourseChapter objects to a dictionary keyed by chapter ID
        chapters_dict = {chapter.id: chapter for chapter in chapters}
        return chapters_dict

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
            If any level references chapter IDs that are not defined in the
            course chapters.
        """
        # The get_level_infos function internally calls load_chapters_from_yaml
        course_levels = get_level_infos(self.course_yaml_path)
        # Validate that all chapter references in levels exist in the course chapters
        chapters = self.course_chapters
        for level_name, level in course_levels.items():
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

        return course_levels

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
