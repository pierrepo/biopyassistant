"""Pydantic configuration models used to validate the CLI argument of BioPyAssistant App."""

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


class LLMConfig(BaseModel):
    """Configuration model for Large Language Model (LLM) parameters."""

    provider_name: Literal["openai", "openrouter"] = Field(
        ..., description="Name of the LLM provider, must be 'openai' or 'openrouter'."
    )
    llm_model_name: str = Field(
        ..., description="Default LLM model name to use if none is provided."
    )
    embedding_model_name: str = Field(
        ...,
        description="Name of the embedding model to use for vector representations.",
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

        The returned key depends on the value of `self.provider`:
        - If `provider` is "openai", returns the value of `api_key_openai`.
        - If `provider` is "openrouter", returns the value of `api_key_openrouter`.

        Returns
        -------
        SecretStr
            The API key corresponding to the currently selected provider.
        """
        if self.provider_name == "openai":
            # if self.api_key_openai is None:
            #     msg = "OPENAI_API_KEY not set"
            #     raise ValueError(msg)
            return self.api_key_openai or SecretStr(os.getenv("OPENAI_API_KEY"))
        elif self.provider_name == "openrouter":
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
    # Paths
    css_path: FilePath = Field(
        ..., description="Path to the CSS file for Streamlit styling."
    )
    vector_database_path: DirectoryPath = Field(
        ..., description="Path to the directory where the vector database is stored."
    )
    # Nested configuration for LLM-related parameters
    llm: LLMConfig = Field(
        ..., description="Configuration for Large Language Model parameters."
    )

    # Compute the log path
    @property
    def log_path(self) -> Path:
        """Return the full path for today's log file."""
        log_file = Path("logs") / f"biopyassistant_app_{datetime.now():%Y-%m-%d}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        return log_file

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
