"""CLI application for searching answers in a vectorial database.

This program allows users to search for answers in a textual database based on a given
query text. It utilizes a similarity search algorithm to find relevant documents in the
database and generates responses to the query using an LLM and the retrieved documents
as context.

Usage:
======
    uv run src/query_chatbot.py --query "Your question here"
                                --level "user_level"
                                [--course-yaml "path_to_yaml_file"]
                                [--model "model_name"]
                                [--provider-llm "provider_llm_name"]
                                [--db-path "database_path"]
                                [--embedding-model "embedding_model"]
                                [--provider-emb "provider_embeddings_name"]
                                [--include-metadata]

Arguments:
==========
    "Your question here" : The query text for which you want to search for answers.
    "user_level" : The user level used to adapt model responses.
                   It can be one of the following:
                   "beginner", "intermediate", "advanced".


    Options:
    ========
    --course-yaml (Path):
            Path to the YAML file defining the course chapters and student levels.
            The YAML should include chapter names, titles, source Markdown paths, and
            processed file paths.
            Default: "data/chapters_and_levels.yaml"

    --model (str):
            The name of the model to be used for generating responses.
            Default: "gpt-4o"

    --provider-llm (str):
            Name of the LLM model provider to use.
            It can be either "openai" or "openrouter".
            Default: "openai"

    --db-path (str):
            File path to the Chroma database containing the context embeddings.
            Default: "chroma_db"

    --embedding-model (str):
            Name of the embedding model to use.
            This should match the embedding model used
            to create the Chroma database.
            Default: "text-embedding-3-large"

    --provider-emb (str):
            Name of the embeddings model provider to use.
            It can be either "openai" or "openrouter".
            Default: "openai"

    --include-metadata (bool):
            Optional flag to specify whether to include metadata in the response.
            If provided, metadata will be included; otherwise, it will be excluded.
            Default: metadata is excluded

Example:
========
    uv run src/query_chatbot.py --query "D'où vient le nom Python ?" \
        --level "beginner" --model "gpt-4o" \
        --course-yaml "data/chapters_and_levels.yaml" \
        --provider-llm "openai" --db-path "chroma_db" \
        --provider-emb "openai" --embedding-model "text-embedding-3-large" \
        --include-metadata

This command will search for answers to the query "Qu'est-ce que Python ?" from a
beginner user in the Chroma database located at "data/chroma_db"
using the "text-embedding-3-large" embedding model from the "openai" provider,
The answer will be generated using the "gpt-4o" model from the "openai" provider,
and the response will include metadata from the relevant documents.
"""

import os
import secrets
from datetime import datetime
from pathlib import Path
from time import perf_counter

import click
import loguru
import tiktoken
import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from create_database import create_embeddings_function
from logger import create_logger
from models.course import CourseLevel

MSGS_QUERY_NOT_RELATED = [
    (
        "Je suis désolé, je ne peux pas répondre à cette question. "
        "Mon domaine d'expertise est la programmation Python. "
        "N'hésite pas à me poser des questions liées à ce sujet,"
        "je serai ravi de t'aider."
    ),
    (
        "Désolé, je suis un assistant pour l'apprentissage de la programmation Python. "
        "Je ne suis pas en mesure de répondre à cette question."
    ),
    (
        "Je ne suis pas sûr de pouvoir répondre à cette question, car elle ne semble "
        "pas être liée à la programmation Python. Si tu as des questions sur Python, "
        "n'hésite pas à me les poser, je serai heureux de t'aider !"
    ),
]
MSGS_QUERY_OUT_OF_SCOPE_LEVEL = [
    (
        "Cette question fait référence à des notions qui ne sont pas encore abordées "
        "dans ce cours."
    ),
    (
        "Cette notion n'est pas encore abordée à votre niveau actuel "
        "et fait partie de la suite du programme."
    ),
    (
        "Cette question fait référence à des notions qui dépassent "
        "le cadre du niveau actuel de votre formation."
    ),
]


def get_level_infos(
    course_yaml: Path, logger: "loguru.Logger" = loguru.logger
) -> dict[str, dict]:
    """
    Load all user level information from a YAML file.

    Parameters
    ----------
    course_yaml : Path
        Path to the YAML file defining course levels.
    logger : loguru.Logger
        Logger for logging messages.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping level name to its info:
        {
            "name": str,
            "display_name": str,
            "comment": str,
            "prompt_path": str,
            "chapters": list[str]
        }
    """
    try:
        with course_yaml.open("r", encoding="utf-8") as f:
            course_data = yaml.safe_load(f) or {}

    except FileNotFoundError:
        logger.error(f"YAML file not found: {course_yaml}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        return {}

    levels = course_data.get("levels", [])
    if not levels:
        logger.warning("No levels defined in YAML file.")
        return {}

    level_infos = {}
    for level in levels:
        name = level.get("name")
        if not name:
            logger.warning("Level without a name found, skipping.")
            continue
        # Construct the level info dictionary
        try:
            level = {
                "name": name,
                "display_name": level.get("display_name"),
                "comment": level.get("comment"),
                "prompt_path": Path(level.get("prompt_path")),
                "chapters": [str(ch) for ch in level.get("chapters")],
            }
        except KeyError as exc:
            logger.warning(
                f"Level '{name}' is missing required field {exc!s}, skipping."
            )
            continue
        # Validate the level info using the CourseLevel Pydantic model
        try:
            validated_level = CourseLevel.model_validate(level)
            level_infos[name] = validated_level
        except ValidationError as exc:
            logger.warning(
                f"Level '{name}' has invalid field value: {exc!s}, skipping."
            )
            continue
    return level_infos


def get_user_level_data(
    user_level: str, course_yaml: Path, logger: "loguru.Logger" = loguru.logger
) -> dict:
    """
    Retrieve user level information, including relevant chapters and prompt file.

    Parameters
    ----------
    user_level : str
        The identifier of the user's level (e.g., 'beginner').
    course_yaml : Path
        Path to the YAML file defining course levels.
    logger : loguru.Logger
        Logger for messages.

    Returns
    -------
    dict
        Dictionary containing:
            - "chapters": list[str] of chapter IDs relevant to the user level
            - "prompt_file": str path to the prompt template

    Raises
    ------
    SystemExit
        Exits with code 1 if the specified user level is not found in the YAML file.
    """
    # Load all levels
    level_infos = get_level_infos(course_yaml, logger)

    # Retrieve the specific user level info
    user_info = level_infos.get(user_level)
    if not user_info:
        available_levels = ", ".join(level_infos.keys()) or "None"
        logger.error(
            f"Failed to retrieve user level '{user_level}' from YAML. "
            f"Available levels: {available_levels}. Exiting."
        )
        raise SystemExit(1)
    chapters = user_info.chapters
    prompt_path = user_info.prompt_path

    return {"chapters": chapters, "prompt_path": prompt_path}


def load_database(
    vector_db_path: str,
    embedding_model: str,
    provider_embeddings_name: str,
    logger: "loguru.Logger" = loguru.logger,
) -> Chroma:
    """Prepare the vector database.

    Parameters
    ----------
    vector_db_path : str
        The file path to the Chroma database containing the context embeddings.
    embedding_model : str
        The name of the embedding model to use for loading the database.
    provider_embeddings_name : str
        The name of the embeddings model provider to use,
        needed to create the appropriate
        embedding function for loading the database.
    logger: "loguru.Logger"
        Logger for logging messages.

    Returns
    -------
        Chroma: The prepared vector database.
    """
    logger.info("Loading the vector database...")
    logger.debug(f"Vector database path: {vector_db_path}")
    logger.debug(f"Embedding model: {embedding_model} from {provider_embeddings_name}")
    # Define the embedding function to use for loading the database
    embedding_function = create_embeddings_function(
        embedding_model, provider_embeddings_name
    )
    # Load the database from the specified directory
    vector_db = Chroma(
        persist_directory=vector_db_path, embedding_function=embedding_function
    )
    # Count the number of chunks in the database
    nb_chunks = vector_db._collection.count()
    logger.info(f"Chunks in the database: {nb_chunks}")
    logger.success("Vector database prepared successfully.")
    return vector_db


def search_similarity_in_database(
    vector_db: Chroma,
    user_query: str,
    nb_chunks: int = 3,
    score_threshold: float = 0.35,
    logger: "loguru.Logger" = loguru.logger,
) -> list[Document]:
    """Search for relevant documents in the database based on the query text.

    Parameters
    ----------
    vector_db : Chroma
        The textual database to search.
    user_query : str
        The query text.
    nb_chunks : int
        The number of top matching documents to retrieve.
    score_threshold : float
        The relevance score threshold for filtering the results.
    logger : "loguru.Logger"
        Logger for logging messages.

    Returns
    -------
    relevant_chunks : list
        List of relevant documents found in the database.
    """
    logger.info("Searching for relevant documents in the database...")
    # Define the retriever
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": nb_chunks,
            "score_threshold": score_threshold,
        },
    )
    # Perform a similarity search
    relevant_chunks = retriever.invoke(user_query)

    # Display information about the relevant chunks
    for chunk in relevant_chunks:
        logger.debug(f"Chunk ID: {chunk.id}")
        logger.debug(f"Chapter name: {chunk.metadata['chapter_name']}")
        logger.debug(f"URL: {chunk.metadata['url']}")
        logger.debug(f"Number of tokens: {chunk.metadata['nb_tokens']}")
        logger.debug("Chunk content:")
        for line in chunk.page_content.splitlines():
            # Split into sentences for better readability
            for sentence in line.split(". "):
                logger.debug(f"{sentence}")
        logger.debug("--------------------------------------")
    logger.success(
        f"Retrieval completed with {len(relevant_chunks)} relevant chunks found."
    )
    return relevant_chunks


def calculate_nb_tokens(text: str) -> int:
    """Calculate the number of tokens in a text.

    Parameters
    ----------
    text : str
        The text for which to calculate the number of tokens.

    Returns
    -------
    int
        The number of tokens in the text.
    """
    # Tokenize the text
    encoding = tiktoken.get_encoding("cl100k_base")
    nb_tokens = len(encoding.encode(text))

    return nb_tokens


def generate_answer(
    query: str,
    user_level: str,
    provider_llm_name: str,
    chat_history: str | None,
    relevant_chunks: list,
    level_relevant_chapter_ids: list[str],
    model_name: str,
    prompt_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> tuple[str, int, int]:
    """Generate an answer to the user query.

    Parameters
    ----------
    query : str
        The user query.
    user_level : str
        The user level, used for logging purposes.
    provider_llm_name : str
        The name of the LLM model provider to use.
    chat_history : str | None
        The contextualized chat history, for IU interface users,
        or None for non-IU users.
    relevant_chunks : list
        List of relevant documents from the database.
    level_relevant_chapter_ids : list[str]
        List of chapter IDs relevant to the user level, used for logging purposes.
    model_name : str
        The name of the OpenAI model to use for generating the answer.
    prompt_path : Path
        The file path to the text file containing the prompt template.
    logger : loguru.Logger
        Logger for logging messages.

    Returns
    -------
    answer : str
        The answer generated by the model.
    nb_tokens_prompt : int
        The number of tokens in the prompt.
    nb_tokens_answer : int
        The number of tokens in the answer.
    """
    # Format the relevant documents for the model
    context = "\n\n".join(
        [
            f"Document {chunk.id} : {chunk.page_content}"
            for i, chunk in enumerate(relevant_chunks)
        ]
    )
    # Load environment variables
    load_dotenv()
    # Define the model
    if provider_llm_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        chat_model = ChatOpenAI(model=model_name, api_key=api_key)
    # For OpenRouter, we don't have a specific class in LangChain,
    # but we can use the ChatOpenAI class with the appropriate base URL and API key.
    # Doc: https://openrouter.ai/docs/guides/community/langchain
    elif provider_llm_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        chat_model = ChatOpenAI(
            model=model_name, api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
    # Retrieve the prompt template
    prompt_template_content = Path(prompt_path).read_text(encoding="utf-8")
    # Define the prompt template
    answer_prompt = ChatPromptTemplate.from_template(prompt_template_content)
    # Define the chained prompt
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    # Input data for the prompt
    input_data = {
        "contexte": context,
        "question": query,
        "chat_history": chat_history,
    }
    # Generate the answer
    answer = answer_chain.invoke(input_data)
    logger.debug(f"Question: {query}")
    logger.debug(f"Prompt path: {prompt_path}")
    logger.debug(f"LLM model used: {model_name} from {provider_llm_name}")
    logger.debug("Answer generated by the LLM:")
    for line in answer.splitlines():
        for sentence in line.split(". "):
            logger.debug(f"{sentence}")
    # Fill the prompt with the input data
    filled_prompt = answer_prompt.format(**input_data)
    nb_tokens_prompt = calculate_nb_tokens(filled_prompt)
    # Calculate the number of tokens in the answer
    nb_tokens_answer = calculate_nb_tokens(answer)

    # Formate the answer if relevant chunks are found but not relevant to the user level
    for chunk in relevant_chunks:
        if chunk.metadata["chapter_id"] not in level_relevant_chapter_ids:
            logger.warning(
                "This question seems to be related to the course "
                "but not relevant to the user level."
            )
            logger.debug(
                f"Chapter `{chunk.metadata['chapter_id']}` "
                f" not in the list of level-relevant chapters for {user_level}: "
                f"{level_relevant_chapter_ids}"
            )
            warning_sentence = secrets.choice(MSGS_QUERY_OUT_OF_SCOPE_LEVEL)
            answer = f"{warning_sentence}\n\n{answer}"
            logger.debug(f"Final answer: {answer.replace('\n\n', ' ')}")
            break

    return answer, nb_tokens_prompt, nb_tokens_answer


def add_metadata_to_answer(
    answer_from_model, relevant_chunks: list[Document], *, iu: bool = False
) -> str:
    """Add metadata to the response.

    Parameters
    ----------
    answer_from_model : str
        The response text predicted by the AI model.
    relevant_chunks : list[Document]
        List of relevant document chunks with their metadata.
    iu : bool
        Flag to specify interface user or not.

    Returns
    -------
    str
        The answer with added metadata.
    """
    # Generate sources string
    sources_set = set()  # Use a set to store unique sources

    # Extract metadata for each relevant chunk
    metadatas = [doc.metadata for doc in relevant_chunks]

    for metadata in metadatas:
        file_name = metadata["file_name"]  # get the file name
        chapter_name = metadata["chapter_name"]  # get the chapter name
        section_name = metadata.get(
            "section_name", ""
        )  # get the section name if it exists
        subsection_name = metadata.get(
            "subsection_name", ""
        )  # get the subsection name if it exists
        subsubsection_name = metadata.get(
            "subsubsection_name", ""
        )  # get the subsubsection name if it exists
        url = metadata.get("url", "")  # get the URL if it exists

        # Determine the most detailed section available
        detailed_section = subsubsection_name or subsection_name or section_name

        if not iu:
            # Construct the source string + URL
            if file_name.startswith("annexe"):
                source_parts = [f"Annexe **{chapter_name}**"]
            else:
                source_parts = [f"Chapitre **{chapter_name}**"]
            if detailed_section:
                source_parts.append(f", rubrique **{detailed_section}**")
            if url:
                source_parts.append(f"(Lien vers la source : {url})")

        else:
            # Get the chapter url
            chapter_url = url.split("#")[0]
            # Construct the source string with a clickable URL
            if file_name.startswith("annexe"):
                source_parts = [f"Annexe [**{chapter_name}**]({chapter_url})"]
            else:
                source_parts = [f"Chapitre [**{chapter_name}**]({chapter_url})"]
            if detailed_section:
                source_parts.append(f", rubrique [**{detailed_section}**")
                if url:
                    source_parts.append(f"]({url})")

        source = " ".join(source_parts)

        # Add the source to the set
        sources_set.add(source)

    sources_list = list(sources_set)  # cast to join into a string
    sources_text = "\n- ".join(sources_list)
    sources_string = (
        "Pour plus d'informations, je t'invite à consulter les rubriques "
        "suivantes du [cours en ligne](https://python.sdv.u-paris.fr/) :\n"
        f"- {sources_text}"
    )

    # Add the sources to the response
    response_with_metadata = f"{answer_from_model}\n\n{sources_string}"

    return response_with_metadata


def display_answer(
    user_query: str,
    answer: str,
    start_time: float,
    logger: "loguru.Logger",
    nb_tokens_prompt: int,
    nb_tokens_answer: int,
) -> None:
    """Display the results.

    Parameters
    ----------
    user_query : str
        The query from the user.
    answer : str
        The response text with added metadata.
    start_time : float
        The start time of the process, used to calculate elapsed time.
    logger : loguru.Logger
        Logger for logging messages.
    nb_tokens_prompt : int
        The number of tokens in the prompt, used for logging usage statistics.
    nb_tokens_answer : int
        The number of tokens in the answer, used for logging usage statistics.
    """
    print("\n\nQuestion:")
    print(f"{user_query}\n\n")
    print("Réponse:")
    print(f"{answer}\n\n")

    logger.debug("Usage statistics:")
    logger.debug(f"Number of tokens in the prompt: {nb_tokens_prompt}")
    logger.debug(f"Number of tokens in the answer: {nb_tokens_answer}")
    logger.debug(f"Total number of tokens: {nb_tokens_prompt + nb_tokens_answer}")

    elapsed_time = perf_counter() - start_time
    logger.success(
        f"Query chatbot completed successfully in {elapsed_time:.2f} seconds!"
    )


@click.command()
@click.option(
    "--query",
    "user_query",
    type=str,
    required=True,
    help="The query text for which you want to search for answers.",
)
@click.option(
    "--level",
    "user_level",
    type=click.Choice(
        ["beginner", "intermediate", "advanced"],
        case_sensitive=False,
    ),
    required=True,
    help="User level used to adapt model responses.",
)
@click.option(
    "--course-yaml",
    "course_yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("data/chapters_and_levels.yaml"),
    help=(
        "Path to the YAML file defining the course chapters and student levels. "
        "The YAML should include chapter names, titles, source Markdown paths, "
        "and processed file paths."
    ),
)
@click.option(
    "--model",
    "model_name",
    type=str,
    default="gpt-4o",
    show_default=True,
    help="The name or identifier of the model to be used for generating responses.",
)
@click.option(
    "--provider-llm",
    "provider_llm_name",
    type=click.Choice(
        ["openai", "openrouter"],
        case_sensitive=False,
    ),
    default="openai",
    help="Name of the LLM model provider to use.",
)
@click.option(
    "--db-path",
    "database_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="chroma_db",
    show_default=True,
    help="File path to the Chroma database containing the context embeddings.",
)
@click.option(
    "--embedding-model",
    "embedding_model",
    default="text-embedding-3-large",
    type=str,
    help="Name of the embedding model to use."
    "This should match the embedding model used to create the Chroma database.",
)
@click.option(
    "--provider-emb",
    "provider_embeddings_name",
    type=click.Choice(
        ["openai", "openrouter"],
        case_sensitive=False,
    ),
    default="openai",
    help="Name of the embeddings model provider to use.",
)
@click.option(
    "--include-metadata",
    is_flag=True,
    default=False,
    help="Include metadata in the response if this flag is provided.",
)
def interrogate_model(
    user_query: str,
    user_level: str,
    course_yaml: Path,
    model_name: str,
    provider_llm_name: str,
    database_path: str,
    embedding_model: str,
    provider_embeddings_name: str,
    *,
    include_metadata: bool,
) -> None:
    """Interrogate the AI model to search for answers in a vector database."""
    start_time = perf_counter()
    # Set-up the logger
    log_path = (
        f"logs/{datetime.now().strftime('%Y%m%d')}/"
        f"query_chatbot_{datetime.now().strftime('%H:%M:%S')}.log"
    )
    logger = create_logger(log_path)
    logger.info("Starting the command line interface for querying the chatbot...")

    # USER LEVEL INFORMATION RETRIEVAL
    # Retrieve the user level information from the YAML file
    user_infos = get_user_level_data(user_level, course_yaml, logger)
    level_relevant_chapter_ids = user_infos["chapters"]
    prompt_path = user_infos["prompt_path"]

    # CONTEXT RETRIEVAL
    # Load the vector database
    vector_db = load_database(database_path, embedding_model, provider_embeddings_name)
    # Search for relevant documents in the database
    relevant_chunks = search_similarity_in_database(
        vector_db, user_query, logger=logger
    )

    # ANSWER GENERATION
    # Check if there are relevant documents
    if relevant_chunks == []:
        logger.warning(
            "This question does not seem to be related to the course content."
        )
        # Avoids calling the model for queries that are not relevant to the course
        answer = secrets.choice(MSGS_QUERY_NOT_RELATED)
        logger.debug(
            f"Answer generated automatically without calling the model: {answer}"
        )
        nb_tokens_prompt = 0
        nb_tokens_answer = 0
    else:
        # Generate the answer
        answer, nb_tokens_prompt, nb_tokens_answer = generate_answer(
            query=user_query,
            user_level=user_level,
            chat_history=None,
            relevant_chunks=relevant_chunks,
            level_relevant_chapter_ids=level_relevant_chapter_ids,
            model_name=model_name,
            provider_llm_name=provider_llm_name,
            prompt_path=prompt_path,
            logger=logger,
        )

        # ANSWER FORMATTING
        # Display the answer with or without metadata based on the include_metadata flag
        if include_metadata:
            # Add metadata of the top matching documents to the answer
            answer = add_metadata_to_answer(answer, relevant_chunks)

    display_answer(
        user_query, answer, start_time, logger, nb_tokens_prompt, nb_tokens_answer
    )


# MAIN PROGRAM
if __name__ == "__main__":
    interrogate_model()
