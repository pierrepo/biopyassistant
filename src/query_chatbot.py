"""CLI application for searching answers in a vectorial database.

This program allows users to search for answers in a textual database based on a given
query text. It utilizes a similarity search algorithm to find relevant documents in the
database and generates responses to the query using an LLM and the retrieved documents
as context.

Usage:
======
    uv run src/query_chatbot.py --query "Your question here"
                                --level "user_level"
                                [--yaml-path "path_to_yaml_file"]
                                [--model "model_name"]
                                [--provider-llm "provider_llm_name"]
                                [--db-path "database_path"]
                                [--embedding-model "embedding_model"]
                                [--provider-emb "provider_embeddings_name"]
                                [--prompt_path "prompt_path"]
                                [--include-metadata]

Arguments:
==========
    "Your question here" : The query text for which you want to search for answers.
    "user_level" : The user level used to adapt model responses.
                   It can be one of the following:
                        - "debutant" : for beginner users
                        - "intermediaire" : for intermediate users
                        - "avance" : for advanced users


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
            Default: "data/chroma_db"

    --embedding-model (str):
            Name of the embedding model to use.
            This should match the embedding model used
            to create the Chroma database.
            Default: "text-embedding-3-large"

    --provider-emb (str):
            Name of the embeddings model provider to use.
            It can be either "openai" or "openrouter".
            Default: "openai"

    --prompt_path (Path):
            File path to the text file containing the prompt template.
            Default: "prompts/zero_shot.txt"

    --include-metadata (bool):
            Optional flag to specify whether to include metadata in the response.
            If provided, metadata will be included; otherwise, it will be excluded.
            Default: metadata is excluded

Example:
========
    uv run src/query_chatbot.py --query "D'où vient le nom Python ?" \
        --level "debutant" --model "gpt-4o" \
        --course-yaml "data/chapters_and_levels.yaml" \
        --provider-llm "openai" --db-path "chroma_db" \
        --provider-emb "openai" --embedding-model "text-embedding-3-large" \
        --prompt_path "prompts/zero_shot.txt" --include-metadata

This command will search for answers to the query "Qu'est-ce que Python ?" from a
begginer user in the Chroma database located at "data/chroma_db"
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

from create_database import create_embeddings_function
from logger import create_logger

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
]
MSGS_QUERY_OUT_OF_SCOPE_LEVEL = [
    (
        "Cette question fait référence à des notions qui ne sont pas encore abordées "
        "dans ce cours. Je te conseille de te concentrer d`abord sur les chapitres "
        "actuellement au programme."
    ),
    (
        "Je suis désolé, mais cette question semble faire référence à des notions qui "
        "ne sont pas encore couvertes dans le cours. Je te recommande de suivre les "
        "chapitres dans l'ordre pour une meilleure compréhension."
    ),
]


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


def get_level_relevant_chapter_ids(
    user_level: str, course_yaml: Path, logger: "loguru.Logger" = loguru.logger
) -> list[str]:
    """Get the list of chapter IDs relevant to the user level.

    Parameters
    ----------
    user_level : str
        The user level used to adapt model responses.
    course_yaml : Path
        The file path to the YAML file defining the course chapters and student levels.
    logger: "loguru.Logger"
        Logger for logging messages.

    Returns
    -------
        list[str]: List of chapter IDs relevant to the user level.
    """
    logger.info("Getting the list of chapter IDs relevant to the user level...")
    with open(course_yaml, encoding="utf-8") as f:
        course_data = yaml.safe_load(f)
    # Get the list of chapter IDs relevant to the user level
    levels = course_data.get("levels", [])
    if not levels:
        logger.warning("No levels defined in YAML file.")
        return []

    for level in levels:
        if level.get("name") == user_level:
            chapter_ids = [str(ch_id) for ch_id in level.get("chapters", [])]
            logger.success(f"Chapter IDs for level '{user_level}': {chapter_ids}")
            return chapter_ids

    logger.warning(f"User level '{user_level}' not found in YAML file.")
    return []


def search_similarity_in_database(
    vector_db: Chroma,
    user_query: str,
    level_relevant_chapter_ids: list[str],
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
    level_relevant_chapter_ids : list[str]
        List of chapter IDs relevant to the user level.
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

    # Create a filter dictionary to restrict the search to user-level relevant chapters
    level_relevant_chapter_ids_filter_dict = {
        "chapter_id": {"$in": level_relevant_chapter_ids}
    }

    # Define the retriever
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": nb_chunks,
            "score_threshold": score_threshold,
            "filter": level_relevant_chapter_ids_filter_dict,
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
            logger.debug(f"{line}")
    logger.success(
        f"Retrieval completed with {len(relevant_chunks)} relevant chunks found."
    )
    return relevant_chunks


def generate_answer_without_model(
    vector_db: Chroma,
    user_query: str,
    logger: "loguru.Logger" = loguru.logger,
    nb_chunks: int = 3,
    score_threshold: float = 0.35,
) -> str:
    """Generate an answer to the user query without calling the model.

    This function is used when no relevant documents are found in the database for the
    user query. It displays personalized messages to the user based on whether the
    query is not related to the user level or not related to the course at all.


    Parameters
    ----------
    vector_db : Chroma
        The textual database to search.
    user_query : str
        The query text.
    logger : loguru.Logger
        Logger for logging messages.
    nb_chunks : int
        The number of top matching documents to retrieve.
    score_threshold : float
        The relevance score threshold for filtering the results.

    Returns
    -------
    response : str
        The response message to be displayed to the user.
    """
    logger.debug(f"Question: {user_query}")
    # Define the retriever without level filter
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": nb_chunks,
            "score_threshold": score_threshold,
        },
    )
    # Perform a similarity search
    relevant_chunks = retriever.invoke(user_query)
    # If there are some relevant chunks but they are not related to the user level
    if relevant_chunks != []:
        logger.warning(
            "This question seems to be related to the course "
            "but not relevant to the user level."
        )
        response = secrets.choice(MSGS_QUERY_OUT_OF_SCOPE_LEVEL)
        logger.debug(
            f"Answer generated automatically without calling the model: {response}"
        )
    else:
        logger.warning(
            "This question does not seem to be related to the course content."
        )
        response = secrets.choice(MSGS_QUERY_NOT_RELATED)
        logger.debug(
            f"Answer generated automatically without calling the model: {response}"
        )
    return response


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
    provider_llm_name: str,
    chat_history: str | None,
    relevant_chunks: list,
    model_name: str,
    prompt_path: Path,
    logger: "loguru.Logger",
) -> tuple[str, int, int]:
    """Generate an answer to the user query.

    Parameters
    ----------
    query : str
        The user query.
    provider_llm_name : str
        The name of the LLM model provider to use.
    chat_history : str | None
        The contextualized chat history, for IU interface users,
        or None for non-IU users.
    relevant_chunks : list
        List of relevant documents from the database.
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
        chat_model = ChatOpenAI(model=model_name, openai_api_key=api_key)
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
    logger.debug(f"Answer generated by the model: {answer}")
    # Fill the prompt with the input data
    filled_prompt = answer_prompt.format(**input_data)
    nb_tokens_prompt = calculate_nb_tokens(filled_prompt)
    # Calculate the number of tokens in the answer
    nb_tokens_answer = calculate_nb_tokens(answer)

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
        ["debutant", "intermediaire", "avance"],
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
    "--prompt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="prompts/zero_shot.txt",
    show_default=True,
    help="File path to the text file containing the prompt template.",
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
    prompt_path: Path,
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
    # Log the user query and parameters
    logger.info("Starting the command line interface for querying the chatbot...")
    logger.debug(f"User query: {user_query}")
    logger.debug(f"User level: {user_level}")
    logger.debug(f"YAML path: {course_yaml}")
    logger.debug(f"Model name: {model_name}")
    logger.debug(f"Provider LLM name: {provider_llm_name}")
    logger.debug(f"Database path: {database_path}")
    logger.debug(f"Embedding model: {embedding_model}")
    logger.debug(f"Provider embeddings name: {provider_embeddings_name}")
    logger.debug(f"Prompt path: {prompt_path}")
    logger.debug(f"Include metadata: {include_metadata}")

    # CONTEXT RETRIEVAL
    # Load the vector database
    vector_db = load_database(database_path, embedding_model, provider_embeddings_name)
    # Get the list of chapter IDs relevant to the user level
    level_relevant_chapter_ids = get_level_relevant_chapter_ids(
        user_level, course_yaml, logger
    )
    # Search for relevant documents in the database
    relevant_chunks = search_similarity_in_database(
        vector_db, user_query, level_relevant_chapter_ids, logger=logger
    )

    # ANSWER GENERATION
    # Check if there are relevant documents
    if relevant_chunks == []:
        # Avoids calling the model for queries that are not relevant to the course
        answer = generate_answer_without_model(vector_db, user_query)
        nb_tokens_prompt = 0
        nb_tokens_answer = 0
    else:
        # Generate the answer
        answer, nb_tokens_prompt, nb_tokens_answer = generate_answer(
            query=user_query,
            chat_history=None,
            relevant_chunks=relevant_chunks,
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
