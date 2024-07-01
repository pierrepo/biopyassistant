"""CLI application for searching answers in a vectorial database.

This program allows users to search for answers in a textual database based on a given query text. 
It utilizes a similarity search algorithm to find relevant documents in the database and generates 
responses to the query using an OpenAI model.

Usage:
======
    python src/query_chatbot.py --query "Your question here"  [--model "model_name"]
                                                              [--include-metadata]                                
                                                           
Arguments:
==========
    "Your question here" : The query text for which you want to search for answers.

    Options:
    ========
    --model "model_name" : The name or identifier of the model to be used for generating responses.
                           (Default model: OPENAI_MODEL_NAME)
    
    --include-metadata : Optional flag to specify whether to include metadata in the response.
                         If provided, metadata will be included; otherwise, it will be excluded.
                         (Default: metadata is excluded)

Example:
========
    python src/query_chatbot.py --query "D'où vient le nom Python ?" --model "gpt-4o" --include-metadata

This command will search for answers to the query "Qu'est-ce que Python ?" in the vectorial Chroma database using the "gpt-4o" model.
And it will include metadata in the response.
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import re
import sys
import random
import argparse
from typing import Tuple, Union, List

import tiktoken
from loguru import logger
from openai import OpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate


# CONSTANTS
CHROMA_PATH = "chroma_db"
OPENAI_MODEL_NAME = "gpt-4o"
PYTHON_LEVEL = "intermédiaire"
EMBEDDING_MODEL = "text-embedding-3-large"

PROMPT_TEMPLATE = """
Tu es un assistant pour les tâches de question-réponse des étudiants dans un cours de programmation Python.
Tu dois fournir des réponses à leurs questions basées sur les supports de cours.
Utilise les morceaux de contexte suivants pour répondre à la question. 
La discussion précédente est également fournie pour t'aider à comprendre le contexte de la question, mais tu ne dois pas l'utiliser pour répondre à la question.

Discussion précédente :
{chat_history}

Question : "{question}"

Contexte : 
"{contexte}"

Répond à la question de manière claire et concise en français de manière adapté à un niveau {niveau_python} en programmation.
La réponse doit être facile à comprendre pour les étudiants.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
Si tu as besoin de plus d'informations, tu dois le demander.
Si tu as besoin de clarifier la question, tu dois le demander.
"""

MSGS_QUERY_NOT_RELATED = [
    "Je suis désolé, je ne peux pas répondre à cette question. Mon domaine d'expertise est la programmation Python. N'hésitez pas à me poser des questions liées à ce sujet, je serai ravi de vous aider.",
    "Désolé, je suis un assistant pour les tâches de question-réponse dans un cours de programmation Python. Je ne suis pas en mesure de répondre à des questions.",
    "Je suis désolé, je ne suis pas sûr de comprendre votre question. Pouvez-vous reformuler votre question en utilisant des termes plus simples ?",
    "Je suis un assistant pour les tâches de question-réponse dans un cours de programmation Python. Je ne suis pas en mesure de répondre à des questions en dehors de ce domaine. Pouvez-vous poser une question liée à la programmation Python ?",
]


# FUNCTIONS
def check_openai_model_validity(model_name):
    # Get the list of available
    models = OpenAI().models.list()
    # Extract GPT models
    models_gpt = [model.id for model in models if "gpt" in model.id]

    # Check if the model name is valid
    if model_name in models_gpt:
        return True
    else:
        return False


def get_args() -> Tuple[str, str, bool]:
    """Parse the command line arguments.

    Returns
    -------
    Tuple[str, str, bool]
        A tuple containing the query, the model name and a flag to include metadata.
    """
    logger.info("Parsing the command line arguments.")
    parser = argparse.ArgumentParser()  # Create a parser object
    # Add arguments to the parser
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="The query text for which you want to search for answers.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OPENAI_MODEL_NAME,
        help="The name or identifier of the model to be used for generating responses.",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        default=False,
        help="Flag to specify whether to include metadata in the response. If provided, metadata will be included.",
    )
    # Parse the command line arguments
    args = parser.parse_args()

    # Checks
    # query is required
    if args.query == "":
        logger.error("Please provide a query")
        sys.exit(1)
    # model name validity
    if not check_openai_model_validity(args.model):
        logger.error(f"The model {args.model} is not valid.")
        sys.exit(1)

    logger.info(f"Query : {args.query}")
    logger.info(f"Model name: {args.model}")
    logger.info(f"Include metadata: {args.include_metadata}")
    logger.success("Command line arguments parsed successfully.\n")

    return args.query, args.model, args.include_metadata


def load_database(vector_db_path: str) -> Tuple[Chroma, int]:
    """Prepare the vector database.

    Returns
    -------
        Chroma: The prepared vector database.
        int: The number of chunks in the database.
    """
    logger.info("Loading the vector database.")
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
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
            formatted_history.append(HumanMessage(content=human))
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
        logger.success("Chat history formatted successfully with 0 entries.\n")
        return chat_history


def contextualize_question(chat_history_formatted: list[Union[HumanMessage, AIMessage]]
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
    logger.success("Contextualized query constructed successfully.\n")

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
    query: str, chat_context: str, relevant_chunks: list, model_name: str, logger_flag: bool = True
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
    chat_model = ChatOpenAI(model=model_name)
    # Define the prompt template
    answer_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Define the chained prompt
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    # Input data for the prompt
    input_data = {
        "contexte": relevant_chunks,
        "niveau_python": PYTHON_LEVEL,
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
    sources_string = (
        f"Pour plus d'informations, consultez les sources suivantes :\n- {sources_text}"
    )

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


def interrogate_model() -> None:
    """Interrogate the AI model to search for answers in a vector database."""
    # Load the query text from the command line arguments
    user_query, model_name, include_metadata = get_args()

    # CONTEXT RETRIEVAL
    # Load the vector database
    vector_db = load_database(CHROMA_PATH)[0]
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
        answer = generate_answer(query=user_query, chat_context=None, relevant_chunks=relevant_chunks_formatted, model_name=model_name)
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
