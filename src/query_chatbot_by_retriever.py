"""CLI application for searching answers in a vectorial database.

This program allows users to search for answers in a textual database based on a given query text. 
It utilizes a similarity search algorithm to find relevant documents in the database and generates 
responses to the query using an OpenAI model.

Usage:
======
    python src/query_chatbot.py --query "Your question here"  [--db-path "path"]
                                                              [--model "model_name"]
                                                              [--python-level "level"] 
                                                              [--include-metadata]
                                                           


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
    --python-level "level" : Optional argument to specify the proficiency level in Python.
                              If provided, it should be one of: "beginner", "intermediate", or "advanced".
                              (Default: "intermediate")
    --include-metadata : Optional flag to specify whether to include metadata in the response.
                         If provided, metadata will be included; otherwise, it will be excluded.
                         (Default: metadata is excluded)

Example:
========
    python src/query_chatbot.py --query "D'où vient le nom Python ?" --model "gpt-3.5-turbo" --python-level "beginner" --include-metadata --db-path "chroma_db"

This command will search for answers to the query "Qu'est-ce que Python ?" in the vector database located at "chroma_db" using the "gpt-3.5-turbo" model.
The response will be generated for a beginner level in Python and will include metadata.
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
import random
import argparse
from typing import Tuple, Union, List, Dict

import tiktoken
from loguru import logger
from openai import OpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# CONSTANTS
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"

PROMPT_TEMPLATE = """
Tu es un assistant pour les tâches de question-réponse des étudiants dans un cours de programmation Python.
Tu dois fournir des réponses à leurs questions basées sur les supports de cours.
Utilise les morceaux de contexte suivants pour répondre à la question.

Question : "{question}"

Contexte : "{contexte}"


Répond à la question de manière claire et concise en français de manière adapté à un niveau {niveau_python} en programmation.
La réponse doit être facile à comprendre pour les étudiants.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
Si tu as besoin de plus d'informations, tu peux le demander.
Si tu as besoin de clarifier la question, tu peux le demander.
"""

PROMPT_REFORMULATE_QUESTION = """
Étant donné un historique de chat et la dernière question de l'utilisateur qui pourrait faire référence au contexte
de l'historique de chat, formule une question autonome qui peut être comprise sans l'historique de chat. 
NE RÉPOND PAS à la question, reformule-la simplement si nécessaire et sinon, retourne-la telle quelle.
"""

MSGS_QUERY_NOT_RELATED = [
    "Je suis désolé, je ne peux pas répondre à cette question. Mon domaine d'expertise est la programmation Python. N'hésitez pas à me poser des questions liées à ce sujet, je serai ravi de vous aider.",
    "Désolé, je suis un assistant pour les tâches de question-réponse dans un cours de programmation Python. Je ne suis pas en mesure de répondre à des questions.",
    "Je suis désolé, je ne suis pas sûr de comprendre votre question. Pouvez-vous reformuler votre question en utilisant des termes plus simples ?",
    "Je suis un assistant pour les tâches de question-réponse dans un cours de programmation Python. Je ne suis pas en mesure de répondre à des questions en dehors de ce domaine. Pouvez-vous poser une question liée à la programmation Python ?"
]


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


def get_args() -> Tuple[str, str, str, bool, str]:
    """Parse the command line arguments.

    Returns
    -------
    Tuple[str, str, str, bool, str]
        A tuple containing the query text, model name, python level, a boolean indicating whether to include metadata, and the path to the vector database.
    """
    logger.info("Parsing the command line arguments.")
    parser = argparse.ArgumentParser() # Create a parser object
    # Add arguments to the parser
    parser.add_argument("--query", type=str, default="",
                        help="The query text for which you want to search for answers.")
    parser.add_argument("--model", type=str, default=OPENAI_MODEL_NAME,
                        help="The name or identifier of the model to be used for generating responses.")
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
    logger.info(f"Python level: {args.python_level}")
    logger.info(f"Include metadata: {args.include_metadata}")
    logger.info(f"Database path: {args.db_path}")
    logger.success("Command line arguments parsed successfully.\n")

    return args.query, args.model, args.python_level, args.include_metadata, args.db_path


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


def search_similarity_in_database(vector_db : Chroma, user_query : str, nb_chunks: int = 3, score_threshold: float = 0.35) -> List[Document]:
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

    Returns
    -------
    relevant_chunks : list
        List of relevant documents found in the database.
    """
    logger.info("Searching for relevant documents in the database.")
    # Define the retriever
    retriever = vector_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nb_chunks, "score_threshold": score_threshold})

    # Perform a similarity search with relevance scores
    relevant_chunks = retriever.invoke(user_query)

    # Display information about the relevant chunks
    for chunk in relevant_chunks:
        logger.info(f"Chunk ID: {chunk.metadata['id']}")
        logger.info(f"Number of tokens: {chunk.metadata['nb_tokens']}")
        logger.info(f"Content: {chunk.page_content[:20]}...")
    

    logger.success("Search completed successfully.\n")

    return relevant_chunks


def format_chat_history(chat_history: list[Tuple[str, str]] = [], len_history: int = 10) -> List[Union[HumanMessage, AIMessage]]:
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
        logger.info("Formatting the chat history.")
        formatted_history = []
        # if chat history is not empty
        if len(chat_history) >0:
            logger.info(f"Chat history is not empty : {len(chat_history)}")
            for human, ai in chat_history[-len_history:]:
                # Append the human and AI messages to the formatted history
                formatted_history.append(HumanMessage(content=human))
                logger.info(f"Human message: {human}")
                formatted_history.append(AIMessage(content=ai))
                logger.info(f"AI message: {ai}")
            logger.success(f"Chat history formatted successfully with {len(formatted_history)} entries.\n")
            return formatted_history
        else: # if chat history is empty
            logger.info("Chat history is empty.")
            logger.success("Chat history formatted successfully with 0 entries.\n")
            return chat_history


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


def contextualize_question(user_query: str, chat_history_formatted: list[Union[HumanMessage, AIMessage]], model_name: str) -> str:
    """Reformulate the chat history into a query.

    Parameters
    ----------
    user_query : str
        The user query.
    chat_history_formatted : list[Union[HumanMessage, AIMessage]]
        The formatted chat history.
    model_name : str
        The name of the model used for the query.

    Returns
    -------
    query_contextualized : str
        The query contextualized with the chat history.
    """
    logger.info("Contextualizing the user query.")
    logger.info(f"User query: {user_query}")

    # Define the model
    chat_model = ChatOpenAI(model=model_name)
    # Define the prompt template
    contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", PROMPT_REFORMULATE_QUESTION),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}")])
    # Define the chained prompt
    contextualize_chain = contextualize_prompt | chat_model | StrOutputParser()
    # Invoke the chained prompt
    contextualized_question = contextualize_chain.invoke({'chat_history': chat_history_formatted, 'question': user_query})
    logger.info(f"Contextualized question: {contextualized_question}")

    logger.success("User query contextualized successfully.\n")
    return contextualized_question


def get_metadata(relevant_chunks : list) -> list[dict]:
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


def generate_answer(query_contextualized: str, relevant_chunks: list, model_name: str, python_level: str) -> str:
    """Generate an answer to the user query.

    Parameters
    ----------
    query_contextualized : str
        The query from the user.
    relevant_chunks : list
        List of relevant documents found in the database.
    model_name : str
        The name of the model used for generating the answer.
    python_level : int
        The Python level of the user.

    Returns
    -------
    answer : str
        The answer generated by the model.
    """
    logger.info("Generating an answer to the user query.")

    # Define the model
    chat_model = ChatOpenAI(model=model_name)
    # Define the prompt template
    answer_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # Define the chained prompt
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    # Input data for the prompt
    input_data = { 'contexte': relevant_chunks, 'niveau_python' : python_level, 'question': query_contextualized}
    # Fill the prompt with the input data
    filled_prompt = answer_prompt.format(**input_data)
    logger.info(f"Filled prompt: {filled_prompt}")
    nb_tokens_prompt = calculate_nb_tokens(filled_prompt)
    logger.info(f"Number of tokens in the prompt: {nb_tokens_prompt}\n")
    # Generate the answer
    answer = answer_chain.invoke(input_data)
    logger.info(f"Answer: {answer}")
    logger.success("Answer generated successfully.\n")

    return answer


def add_metadata_to_answer(answer_from_model, metadatas: list[dict], iu: bool = False) -> str:
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
    logger.info("Adding metadata to the response.")

    # Generate sources string
    sources_set = set()  # Use a set to store unique sources

    for metadata in metadatas:
        chapter_name = metadata['chapter_name'] # get the chapter name
        section_name = metadata.get('section_name', '') # get the section name if it exists
        subsection_name = metadata.get('subsection_name', '') # get the subsection name if it exists
        subsubsection_name = metadata.get('subsubsection_name', '') # get the subsubsection name if it exists
        url = metadata.get('url', '') # get the URL if it exists

        
        if not iu:
            # Construct the source string  + URL
            source_parts = [f"Le chapitre *{chapter_name}*"]
            if section_name:
                source_parts.append(f"la section *{section_name}*")
            if subsection_name:
                source_parts.append(f"la sous-section *{subsection_name}*")
            if subsubsection_name:
                source_parts.append(f"la sous-sous-section *{subsubsection_name}*")
            # Add the URL to the source string 
            source_parts.append(f"(Lien vers la source : {url})")
        
        else:
            # Construct the source string with a clickable URL
            source_parts = [f"[Le chapitre **{chapter_name}**,"]
            if section_name:
                source_parts.append(f"la section **{section_name}**,")
            if subsection_name:
                source_parts.append(f"la sous-section **{subsection_name}**,")
            if subsubsection_name:
                source_parts.append(f"la sous-sous-section **{subsubsection_name}**")
            # Add the URL to the source string 
            source_parts.append(f"]({url})")

        source = " ".join(source_parts)

        # Add the source to the set
        sources_set.add(source)

    sources_list = list(sources_set) # cast to join into a string
    sources_text = "\n- ".join(sources_list)
    sources_string = f"Pour plus d'informations, consultez les sources suivantes :\n- {sources_text}"

    # Add the sources to the response 
    response_with_metadata = f"{answer_from_model}\n\n{sources_string}"

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
    print("Réponse:")
    print(f"{final_response}")
    print("\n\n")

    logger.success("Results displayed successfully.")


def interrogate_model() -> None:
    """Interrogate the AI model to search for answers in a vector database."""
    # Load the query text from the command line arguments
    user_query, model_name, python_level, include_metadata, vector_db_path= get_args()

    # CONTEXT RETRIEVAL
    # Load the vector database
    vector_db = load_database(vector_db_path)[0]
    # Search for relevant documents in the database
    relevant_chunks = search_similarity_in_database(vector_db, user_query)


    # ANSWER GENERATION
    # Check if there are relevant documents
    if relevant_chunks == []:
        # random response betweet responses in MSGS_QUERY_NOT_RELATED
        response = random.choice(MSGS_QUERY_NOT_RELATED)
        print(response)
        sys.exit(0)
    else :
        # Get the metadata of the top matching documents
        metadatas = get_metadata(relevant_chunks)
        # Generate the answer
        answer = generate_answer(user_query, relevant_chunks, model_name, python_level)
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
