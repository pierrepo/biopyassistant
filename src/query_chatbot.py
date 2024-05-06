"""CLI application for searching answers in a vectorial database.

This program allows users to search for answers in a textual database based on a given query text. 
It utilizes a similarity search algorithm to find relevant documents in the database and generates 
responses to the query using an OpenAI model.

Usage:
======
    python src/query_chatbot.py --query "Your question here" [--model "model_name"]
                                                            [--question-type "type"]
                                                            [--python-level "level"] 
                                                            [--include-metadata]
                                                            [--db-path "path"]


Arguments:
==========
    "Your question here" : The query text for which you want to search for answers.

    Options:
    ========
    --db-path "path" : Optional argument to specify the path to the vector database.
                          If provided, the database will be loaded from the specified directory.
                          (Default path: CHROMA_PATH)
    --model "model_name" : The name or identifier of the model to be used for generating responses.
                           (Default model: "gpt-3.5-turbo")
    --question-type "type" : Optional argument to specify the type of question.
                            If provided, it should be one of: "course" or "exercise".
                            (Default: "course")
    --python-level "level" : Optional argument to specify the proficiency level in Python.
                              If provided, it should be one of: "beginner", "intermediate", or "advanced".
                              (Default: "intermediate")
    --include-metadata : Optional flag to specify whether to include metadata in the response.
                         If provided, metadata will be included; otherwise, it will be excluded.
                         (Default: metadata is excluded)

Example:
========
    python src/query_chatbot.py --query "Qu'est-ce que Python ?" --model "gpt-3.5-turbo" --question-type "course" --python-level "beginner" --include-metadata --db-path "chroma_db"

"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import sys
import argparse
from typing import Tuple



import tiktoken
from loguru import logger
from openai import OpenAI
from langchain_core.documents import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# CONSTANTS
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-large"

PROMPT_TEMPLATE_COURSE = """
Tu es un assistant pour les tâches de question-réponse des étudiants dans un cours de programmation Python.
Tu dois fournir des réponses à leurs questions basées sur les supports de cours.
Utilise les morceaux de contexte suivants pour répondre à la question.

Question : "{question}"

Contexte : "{contexte}"


Répond à la question de manière claire et concise en français de manière adapté à un niveau {python_level} en programmation.
La réponse doit être facile à comprendre pour les étudiants.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
Si tu as besoin de plus d'informations, tu peux le demander.
Si tu as besoin de clarifier la question, tu peux le demander.
"""

PROMPT_TEMPLATE_EXERCICE = """
Tu es un assistant pour les tâches de question-réponse des étudiants dans un cours de programmation Python.
Tu dois donner les aider à résoudre des exercices basés sur les supports de cours.
Utilise les morceaux de contexte suivants pour répondre à la question.

Question : "{question}"

Contexte : "{contexte}"

Répond à la question de manière claire et concise en français de manière adapté à un niveau {python_level} en programmation.
Mets en avant les étapes pour résoudre l'exercice.
Ne donne jamais la réponse directement.
La réponse doit être facile à comprendre pour les étudiants.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
Si tu as besoin de plus d'informations, tu peux le demander.
Si tu as besoin de clarifier la question, tu peux le demander.
"""


# FUNCTIONS
def check_openai_model_validity(model_name):
    # Get the list of available 
    models =  OpenAI().models.list()
    # Extract GPT models
    models_gpt = [model.id for model in models if "gpt" in model.id]

    # Check if the model name is valid
    if model_name in models_gpt:
        return True
    else:
        return False


def get_args() -> Tuple[str, str, str, str, bool, str]:
    """Parse the command line arguments.

    Returns
    -------
    Tuple[str, str, str, str, bool, str]
        A tuple containing the query text, model name, python level, a boolean indicating whether to include metadata, and the path to the vector database.
    """
    logger.info("Parsing the command line arguments.")
    parser = argparse.ArgumentParser() # Create a parser object
    # Add arguments to the parser
    parser.add_argument("--query", type=str, default="",
                        help="The query text for which you want to search for answers.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo",
                        help="The name or identifier of the model to be used for generating responses.")
    parser.add_argument("--question-type", type=str, default="Course",
                        help="The type of question. It should be one of: 'Course' or 'Exercice'. (Default: 'Course')")
    parser.add_argument("--python-level", type=str, default="intermediate",
                        help="The proficiency level in Python. It should be one of: 'beginner', 'intermediate', or 'advanced'. (Default: 'intermediate')")
    parser.add_argument("--include-metadata", action="store_true", default=False,
                        help="Flag to specify whether to include metadata in the response. If provided, metadata will be included.")
    parser.add_argument("--db-path", type=str, default=CHROMA_PATH,
                        help="The path to the vector database. If provided, the database will be loaded from the specified directory.")

    # Parse the command line arguments
    args = parser.parse_args()

    # Checks
    # query is required
    if args.query == "":
        logger.error("Please provide a query")
        sys.exit(1)
    # question type should be either "Course" or "Exercice"
    if args.question_type.lower() not in ["course", "exercice"]:
        logger.error("The question type should be either 'Course' or 'Exercice'")
        sys.exit(1)
    # python level should be either "beginner", "intermediate" or "advanced"
    if args.python_level.lower() not in ["beginner", "intermediate", "advanced"]:
        logger.error("The python level should be either 'beginner', 'intermediate' or 'advanced'")
        sys.exit(1)
    # model name validity
    if not check_openai_model_validity(args.model):
        logger.error(f"The model {args.model} is not valid.")
        sys.exit(1)
    # db path should be a valid path
    if not os.path.exists(args.db_path):
        logger.error(f"The database path {args.db_path} is not valid.")
        sys.exit(1)

    logger.info(f"Query text: {args.query}")
    logger.info(f"Model name: {args.model}")
    logger.info(f"Question type: {args.question_type}")
    logger.info(f"Python level: {args.python_level}")
    logger.info(f"Include metadata: {args.include_metadata}")
    logger.info(f"Database path: {args.db_path}")
    logger.success("Command line arguments parsed successfully.\n")

    return args.query, args.model, args.question_type, args.python_level, args.include_metadata, args.db_path


def load_database(vector_db_path: str) -> Tuple[Chroma, int]:
    """Prepare the vector database.

    Returns
    -------
        Chroma: The prepared vector database.
        int: The number of chunks in the database.
    """
    logger.info("Loading the vector database.")
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL) # define the embedding model
    # Load the database from the specified directory
    vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)
    # Count the number of chunks in the database
    nb_chunks = vector_db._collection.count()
    logger.info(f"Chunks in the database: {nb_chunks}")

    logger.success("Vector database prepared successfully.\n")

    return vector_db, nb_chunks


def search_similarity_in_database(db : Chroma, query_text : str) -> list[tuple[Document, float]]:
    """Search the database for relevant documents.

    Parameters
    ----------
    db : Chroma
        The textual database to search.
    query_text : str
        The query text.

    Returns
    -------
        list: A list of top matching documents and their relevance scores.
    """
    logger.info("Searching for relevant documents in the database.")
    # Perform a similarity search with relevance scores
    results = db.similarity_search_with_relevance_scores(query_text, score_threshold=0.40)
    logger.info(f"There are {len(results)} relevant documents found.")

    # Display the number of tokens for each document
    for doc, _score in results:
        logger.info(f"Chunk ID: {doc.metadata["id"]}, Score: {_score}, Number of tokens: {doc.metadata['nb_tokens']}")

    logger.success("Search completed successfully.\n")

    """
    # Print the top matching documents and their relevance scores
    for i, (doc, score) in enumerate(results):
        print(f"Document {i + 1}:")
        print(f"Relevance Score: {score}")
        print(f"Page Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("----")
    """

    return results


def get_metadata(results : list[tuple[Document, float]]) -> list[dict]:
    """Get the metadata of the top matching documents.

    Parameters
    ----------
    results : list
        List of top matching documents and their relevance scores.

    Returns
    -------
        list: List of metadata dictionaries for the top matching documents.
    """
    logger.info("Extracting metadata of the top matching documents.")
    metadatas = [doc.metadata for doc, _score in results]
    logger.success("Metadata extracted successfully.\n")

    return metadatas


def generate_prompt(results : list[tuple[Document, float]], query_text : str, python_level: str, question_type: str) -> str:
    """Generate a prompt for the AI model.

    Parameters
    ----------
    results : list
        List of top matching documents and their relevance scores.
    query_text : str
        The query text.
    python_level : str
        Python proficiency level.
    question_type : str
        Type of question (course or exercice).

    Returns
    -------
        str: The generated prompt.
    """
    logger.info("Generating a prompt for the AI model.")

    # Extract the context text from the top matching documents
    contexts = [doc.page_content for doc, _score in results]

    # Combine context text using appropriate delimiters
    context_text = "\n\n---\n\n".join(contexts)

    # Create a prompt template
    if question_type == "Cours":
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_COURSE)
    else:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_EXERCICE)

    # Fill in the prompt template with the extracted information
    prompt = prompt_template.format(contexte=context_text, question=query_text, python_level=python_level)
    logger.info(f"Prompt: {prompt}")

    # Count the number of tokens in the prompt
    encoding = tiktoken.get_encoding("cl100k_base")
    nb_token_in_prompt = len(encoding.encode(prompt))
    logger.info(f"Number of tokens in the prompt: {nb_token_in_prompt}")
    
    logger.success("Prompt generated successfully.\n")

    return prompt, nb_token_in_prompt


def predict_response(prompt: str, model_name: str, nb_token_in_prompt: int) -> str:
    """Predict a response using an AI model.

    Parameters
    ----------
    prompt: str
        The prompt for the AI model.
    model_name: str
        The name or identifier of the AI model.
    nb_token_in_prompt: int
        The number of tokens in the prompt.

    Returns
    -------
    str
        The predicted response.
    """
    logger.info("Predicting the response using the AI model.")
    """
    # Check the token limit for the model
    model_details = OpenAI().models.retrieve(model_name)
    token_limit = model_details.
    logger.info(f"Token limit for the model {model_name}: {token_limit}, type: {type(token_limit)}")
    if nb_token_in_prompt > token_limit:
        logger.error(f"The prompt has {nb_token_in_prompt} tokens, it's too long for the model {model_name}. The token limit is {token_limit}.")
        sys.exit(1)
    """
    # Initialize the OpenAI model
    model = ChatOpenAI(model= model_name) 

    # Predict the response text
    response_text = model.invoke(prompt).content

    logger.info(f"Response text from the {model}: {response_text}")
    logger.success("Response predicted successfully.\n")

    return response_text


def adding_metadatas_to_response(response_from_model: str, metadatas: list[dict]) -> str:
    """Add metadata to the response.

    Parameters
    ----------
    response_text : str
        The response text predicted by the AI model.
    metadatas : list
        List of metadata dictionaries for the top matching documents.

    Returns
    -------
    str
        The response text with added metadata.
    """
    logger.info("Adding metadata to the response.")

    # Generate sources string
    sources_set = set()  # Use a set to store unique sources

    for metadata in metadatas:
        chapter_name = metadata['chapter_name'] # get the chapter name
        section_name = metadata.get('section_name', '') # get the section name if it exists
        subsection_name = metadata.get('subsection_name', '') # get the subsection name if it exists
        subsubsection_name = metadata.get('subsubsection_name', '') # get the subsubsection name if it exists
        url = metadata.get('url', '') # get the URL if it exists

        # Construct the source string with a clickable URL
        source_parts = [f"Le chapitre *{chapter_name}*"]
        if section_name:
            source_parts.append(f"la section *{section_name}*")
        if subsection_name:
            source_parts.append(f"la sous-section *{subsection_name}*")
        if subsubsection_name:
            source_parts.append(f"la sous-sous-section *{subsubsection_name}*")

        # Add the URL as a clickable link if it exists
        if url:
            source_parts.append(f"(Lien vers la source : {url})")

        source = " ".join(source_parts)

        # Add the source to the set
        sources_set.add(source)

    sources_list = list(sources_set) # cast to join into a string
    sources_text = "\n- ".join(sources_list)
    sources_string = f"Pour plus d'informations, consultez les sources suivantes :\n- {sources_text}"

    # Add the sources to the response 
    response_with_metadata = f"{response_from_model}\n\n{sources_string}"

    return response_with_metadata 



def print_results(query_text: str, final_response: str) -> None:
    """Display the results.

    Parameters
    ----------
    query_text : str
        The query text.
    final_response : str
        The response text with added metadata.
    """
    logger.info("Displaying the results.")

    print("\n\n")
    print("Question:")
    print(f"{query_text}\n")
    print("Réponse:")
    print(f"{final_response}")
    print("\n\n")

    logger.success("Results displayed successfully.")


def interrogate_model() -> None:
    """Interrogate the AI model to search for answers in a vector database."""
    # Load the query text from the command line arguments
    user_query, model_name, question_type, python_level, include_metadata, vector_db_path= get_args()

    # Load the vector database
    vector_db = load_database(vector_db_path)[0]

    # Search for relevant documents in the database
    results = search_similarity_in_database(vector_db, user_query)

    # Check if there are relevant documents
    if not results:
        print(f"Peux-reformuler ou préciser ta question, je n'ai pas trouvé de réponse.")
        sys.exit(0)

    # if there are relevant documents
    # Get the metadata of the top matching documents
    metadatas = get_metadata(results)

    # Generate a prompt for the AI model
    prompt, nb_tokens_in_prompt = generate_prompt(results, user_query, python_level, question_type)

    # Predict the response using the AI model
    response_from_model = predict_response(prompt, model_name, nb_tokens_in_prompt)

    if include_metadata:
        # Add metadata to the response
        final_response = adding_metadatas_to_response(response_from_model, metadatas)

        # Display the results
        print_results(user_query, final_response)

    else:
        # Display the results
        print_results(user_query, response_from_model)


# MAIN PROGRAM
if __name__ == "__main__":
    interrogate_model()

