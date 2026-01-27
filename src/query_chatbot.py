"""CLI application for searching answers in a vectorial database.

This program allows users to search for answers in a textual database based on a given query text.
It utilizes a similarity search algorithm to find relevant documents in the database and generates
responses to the query using an LLM and the retrieved documents as context.

Usage:
======
    python src/query_chatbot.py --query "Your question here"  [--model "model_name"]
                                                              [--prompt_path "prompt_path"]
                                                              [--include-metadata]

Arguments:
==========
    "Your question here" : The query text for which you want to search for answers.

    Options:
    ========
    --model (str) : The name or identifier of the model to be used for generating responses.
                           (Default model: DEFAULT_LLM_MODEL)

    --prompt_path (Path)  : File path to the text file containing the prompt template.
                            (Default: "prompts/few_shot.txt")

    --include-metadata : Optional flag to specify whether to include metadata in the response.
                         If provided, metadata will be included; otherwise, it will be excluded.
                         (Default: metadata is excluded)

Example:
========
    python src/query_chatbot.py --query "D'où vient le nom Python ?" --model "gpt-4o" --prompt_path "prompts/zero_shot.txt" --include-metadata

This command will search for answers to the query "Qu'est-ce que Python ?" in the vectorial Chroma database using the "gpt-4o" model and the zero_shot prompt.
And it will include metadata in the response.
"""

import datetime
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Union

import click
import tiktoken
from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from logger import create_logger

MSGS_QUERY_NOT_RELATED = [
    (
        "Je suis désolé, je ne peux pas répondre à cette question. "
        "Mon domaine d'expertise est la programmation Python. "
        "N'hésite pas à me poser des questions liées à ce sujet, je serai ravi de t'aider."
    ),
    (
        "Désolé, je suis un assistant pour l'apprentissage de la programmation Python. "
        "Je ne suis pas en mesure de répondre à cette question."
    ),
]


def load_database(vector_db_path: str, embedding_model: str) -> Tuple[Chroma, int]:
    """Prepare the vector database.

    Returns
    -------
        Chroma: The prepared vector database.
        int: The number of chunks in the database.
    """
    logger.info("Loading the vector database.")
    embedding_function = OpenAIEmbeddings(
        model=embedding_model
    )  # define the embedding model
    # Load the database from the specified directory
    vector_db = Chroma(
        persist_directory=vector_db_path, embedding_function=embedding_function
    )
    # Count the number of chunks in the database
    nb_chunks = vector_db._collection.count()
    logger.info(f"Chunks in the database: {nb_chunks}")

    logger.success("Vector database prepared successfully.\n")

    return vector_db, nb_chunks


def search_similarity_in_database(
    vector_db: Chroma,
    user_query: str,
    nb_chunks: int = 3,
    score_threshold: float = 0.35,
    logger_flag: bool = True,
) -> List[Document]:
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
    logger_flag : bool
        Flag to indicate whether to log the search results.

    Returns
    -------
    relevant_chunks : list
        List of relevant documents found in the database.
    """
    if logger_flag:
        logger.info("Searching for relevant documents in the database...")

    # Define the retriever
    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": nb_chunks, "score_threshold": score_threshold},
    )

    # Perform a similarity search with relevance scores
    relevant_chunks = retriever.invoke(user_query)

    if logger_flag:
        # Display information about the relevant chunks
        for chunk in relevant_chunks:
            logger.info(f"Chunk ID: {chunk.metadata['id']}")
            logger.info(f"Number of tokens: {chunk.metadata['nb_tokens']}")
            logger.info(f"Content: {chunk.page_content[:20]}...\n")

        logger.success("Search completed successfully.\n")

    return relevant_chunks


def format_relevant_chunks(relevant_chunks: list) -> str:
    """Format the relevant documents for the OpenAI model.

    Parameters
    ----------
    relevant_chunks : list
        List of relevant documents from the database.

    Returns
    -------
    formatted_chunks : str
        The formatted relevant documents.
    """
    logger.info("Formatting the relevant documents.")
    formatted_chunks = ""

    for chunk in relevant_chunks:
        formatted_chunks += f"Chunk ID: {chunk.metadata['id']}\n"
        formatted_chunks += f"Content: {chunk.page_content}\n"
        formatted_chunks += f"Source: {chunk.metadata}\n\n"

    logger.success("Relevant documents formatted successfully.\n")

    return formatted_chunks


def format_chat_history(
    chat_history: list[Tuple[str, str]] = [], len_history: int = 10
) -> List[Union[HumanMessage, AIMessage]]:
    """Format the chat history for the promt template.

    Parameters
    ----------
    chat_history : list[tuple[str, str]], optional
        The chat history to format, by default [].
    len_history : int, optional
        The number of chat history entries to consider, by default 10.

    Returns
    -------
    list[Union[HumanMessage, AIMessage]]
        The formatted chat history.
    """
    logger.info("Formatting the chat history...")
    formatted_history = []
    # Define the pattern to identify and remove metadata
    metadata_pattern = re.compile(
        r"Pour plus d\'informations, consultez les sources suivantes :.*$", re.DOTALL
    )
    # if chat history is not empty
    if len(chat_history) > 0:
        for human, ai in chat_history[-len_history:]:
            # Append the human and AI messages to the formatted history
            if human != None:
                formatted_history.append(HumanMessage(content=human))
            else:
                formatted_history.append(HumanMessage(content=""))
            logger.info(f"Human message: {human}")

            # Remove metadata from AI message
            cleaned_ai = re.sub(metadata_pattern, "", ai).strip()
            formatted_history.append(AIMessage(content=cleaned_ai))
            logger.info(f"AI message (cleaned): {cleaned_ai}")

        logger.success(
            f"Chat history formatted successfully with {len(formatted_history)} entries.\n"
        )
        return formatted_history

    else:  # if chat history is empty
        logger.info("Chat history is empty.")
        logger.success("Chat history formatted successfully with 0 entries.")
        return chat_history


def contextualize_question(
    chat_history_formatted: list[Union[HumanMessage, AIMessage]],
) -> str:
    """Add context to the user query using the chat history.

    Parameters
    ----------
    chat_history_formatted : list[Union[HumanMessage, AIMessage]]
        The formatted chat history.

    Returns
    -------
    chat_context : str
        The contextualized user query.
    """
    logger.info("Contextualizing the user query...")
    chat_context = ""

    # Iterate over the formatted chat history and append to the context
    for i, message in enumerate(chat_history_formatted):
        if isinstance(message, HumanMessage):
            chat_context += f"Question {i // 2 + 1}: {message.content}\n"
        elif isinstance(message, AIMessage):
            chat_context += f"Réponse {i // 2 + 1}: {message.content}\n"

    logger.info(f"Chat context: {chat_context}")
    logger.success("Contextualized query constructed successfully.")

    return chat_context


def get_metadata(relevant_chunks: list) -> list[dict]:
    """Get the metadata of the top matching documents.

    Parameters
    ----------
    relevant_chunks : list
        List of top matching documents and their metadata.

    Returns
    -------
    metadatas : list
        List of metadata dictionaries for the top matching documents.
    """
    logger.info("Extracting metadata of the top matching documents.")
    metadatas = [doc.metadata for doc in relevant_chunks]
    logger.success("Metadata extracted successfully.\n")

    return metadatas


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
    chat_context: str,
    relevant_chunks: list,
    model_name: str,
    prompt_path: Path,
    logger_flag: bool = True,
) -> str:
    """Generate an answer to the user query.

    Parameters
    ----------
    query : str
        The user query.
    chat_context : str
        The contextualized chat history.
    relevant_chunks : list
        List of relevant documents from the database.
    model_name : str
        The name of the OpenAI model to use for generating the answer.
    logger_flag : bool, optional
        Flag to indicate whether to log the output, by default True.

    Returns
    -------
    answer : str
        The answer generated by the model.
    """
    if logger_flag:
        logger.info("Generating an answer to the user query...")

    # Define the model
    if model_name in OPENAI_MODELS:
        chat_model = ChatOpenAI(model=model_name)
    # Retrieve the prompt template
    with open(prompt_path, encoding="utf-8") as f:
        prompt_template_content = f.read()
    # Define the prompt template
    answer_prompt = ChatPromptTemplate.from_template(prompt_template_content)
    # Define the chained prompt
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    # Input data for the prompt
    input_data = {
        "contexte": relevant_chunks,
        "question": query,
        "chat_history": chat_context,
    }
    if logger_flag:
        # Fill the prompt with the input data
        filled_prompt = answer_prompt.format(**input_data)
        logger.info(f"Filled prompt: {filled_prompt}")
        nb_tokens_prompt = calculate_nb_tokens(filled_prompt)
        logger.info(f"Number of tokens in the prompt: {nb_tokens_prompt}\n")
    # Generate the answer
    answer = answer_chain.invoke(input_data)
    if logger_flag:
        logger.success("Answer generated from LLM successfully.\n")

    return answer


def add_metadata_to_answer(
    answer_from_model, metadatas: list[dict], iu: bool = False
) -> str:
    """Add metadata to the response.

    Parameters
    ----------
    answer : str
        The response text predicted by the AI model.
    metadatas : list
        List of metadata dictionaries for the top matching documents.
    iu : bool
        Flag to specify interface user or not.

    Returns
    -------
    str
        The answer with added metadata.
    """
    logger.info("Adding metadata to the response...")

    # Generate sources string
    sources_set = set()  # Use a set to store unique sources

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
    sources_string = f"Pour plus d'informations, je t'invite à consulter les rubriques suivantes du [cours en ligne](https://python.sdv.u-paris.fr/) :\n- {sources_text}"

    # Add the sources to the response
    response_with_metadata = f"{answer_from_model}\n\n{sources_string}"

    logger.info(f"Answer with metadata: {response_with_metadata}")
    logger.success("Metadata added to the response successfully.\n")

    return response_with_metadata


def display_answer(user_query: str, final_response: Union[str, dict]) -> None:
    """Display the results.

    Parameters
    ----------
    user_query : str
        The query from the user.
    final_response : str
        The response text with added metadata.
    """
    logger.info("Displaying the results.")

    print("\n\n")
    print("Question:")
    print(f"{user_query}\n")
    print("Reponse:")
    print(f"{final_response}")
    print("\n\n")

    logger.success("Results displayed successfully.")


@click.command()
@click.option(
    "--query",
    "user_query",
    type=str,
    required=True,
    help="The query text for which you want to search for answers.",
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
    "--provider",
    "provider_name",
    default="openrouter",
    type=str,
    help="Name of the LLM model provider to use.",
)
@click.option(
    "--db-path",
    "database_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="data/chroma_db",
    show_default=True,
    help="File path to the Chroma database containing the context embeddings.",
)
@click.option(
    "--embedding-model",
    "embedding_model",
    default="text-embedding-3-large",
    type=str,
    help="Name of the embedding model to use.",
)
@click.option(
    "--prompt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="prompt/zero_shot.txt",
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
    model_name: str,
    provider_name: str,
    database_path: str,
    embedding_model: str,
    prompt_path: Path,
    include_metadata: bool,
) -> None:
    """Interrogate the AI model to search for answers in a vector database."""
    # Set-up the logger
    log_path = f"logs/query_chatbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = create_logger(log_path)

    # Load the environment variables
    load_dotenv()

    # CONTEXT RETRIEVAL
    # Load the vector database
    vector_db = load_database(database_path, embedding_model)[0]
    # Search for relevant documents in the database
    relevant_chunks = search_similarity_in_database(vector_db, user_query)

    # ANSWER GENERATION
    # Check if there are relevant documents
    if relevant_chunks == []:
        # random response betweet responses in MSGS_QUERY_NOT_RELATED
        response = random.choice(MSGS_QUERY_NOT_RELATED)
        print(response)
        sys.exit(0)
    else:
        # Format the relevant documents for the model
        relevant_chunks_formatted = format_relevant_chunks(relevant_chunks)
        # Get the metadata of the top matching documents
        metadatas = get_metadata(relevant_chunks)
        # Generate the answer
        answer = generate_answer(
            query=user_query,
            chat_context=None,
            relevant_chunks=relevant_chunks_formatted,
            model_name=model_name,
            prompt_path=prompt_path,
        )
        # Calculate the number of tokens in the answer
        logger.info("Calculating the number of tokens in the answer.")
        nb_tokens_answer = calculate_nb_tokens(answer)
        logger.success(f"Number of tokens in the answer: {nb_tokens_answer}\n")

        # ANSWER FORMATTING
        # Add metadata to the answer
        if include_metadata:
            answer_with_metadata = add_metadata_to_answer(answer, metadatas)
            display_answer(user_query, answer_with_metadata)
        else:
            display_answer(user_query, answer)


# MAIN PROGRAM
if __name__ == "__main__":
    interrogate_model()
