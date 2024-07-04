"""Creates Quiz from a chapter of the python course in JSON format using a LLM. 


Usage:
======
    python src/create_quiz.py --chapter [CHAPTER] --quiz-type [QUIZ_TYPE] --python-level [PYTHON_LEVEL]

Arguments:
==========
    --chapter : str
        The chapter of the quiz.        
    --quiz-type : str
        The type of the quiz.
    --python-level : str
        The Python level of the user.

Example:
========
    python src/create_quiz.py --chapter 04_listes --quiz-type QCM --python-level Débutant

This command will create a QCM quiz on the chapter 04_listes for a beginner level student.
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import argparse
from typing import List, Tuple

import re
from loguru import logger
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# MODULE IMPORTS
from query_chatbot import calculate_nb_tokens, OPENAI_MODEL_NAME
from create_database import load_documents, get_file_names


# CONSTANTS
PROCESSED_DATA_PATH = "data/markdown_processed"

GENERATE_QUIZ_PROMPT = """
Tu es un expert en création de quiz sur Python destiné à des étudiants.
Créer un quiz de type {quiz_type} avec 5 questions sur le chapitre suivant : {chapter} pour un étudiant de niveau {level}.
La question doit être au format markdown et doit porter sur le contenu du chapitre fourni et doit être accompagnée de réponses possibles et d'une explication détaillée pour chaque réponse.

Le format de sortie doit seulement etre un JSON comportant les 5 questions de quiz avec les clés suivantes :
- type : le type du quiz (QCM, Vrai ou Faux, Trouver l'erreur dans le code, Trouver la sortie du code, Compléter le code)
- chapter : le chapitre du quiz
- python_level : le niveau Python de l'étudiant
- question : la question posée
- answers : les réponses possibles numérotées par ordre alphabétique (A, B, C, D)
- correct_answer : la réponse correcte avec la lettre correspondante
- explanation : l'explication détaillée de la réponse correcte

Je veux dans explanation, un dictionnaire avec pour chaque réponse possible la lettre correspondante comme clé et comme valeur l'explication de cette réponse.
Cette explication doit commencer par "La réponse ... est incorrecte/correcte car ..." suivi d'une explication détaillée.
Cette explication doit dépendre de la réponse donnée :
- Si la réponse est incorrecte, l'explication doit être détaillée et encourageante.
- Si la réponse est correcte, l'explication doit être brève, informative.

Voici le contenu du chapitre sur lequel tu dois créer le quiz :
{chapter_content}
"""


# FUNCTIONS
def get_args() -> Tuple[str, str, str]:
    """Get the command line arguments.

    Returns
    -------
    Tuple[str, str, str]
        The chapter, quiz_type and the python level of the user.
    """
    logger.info("Parsing command line arguments")
    parser = argparse.ArgumentParser(description="Creates the QCM from a chapter of the python course.")
    parser.add_argument("--chapter", type=str, required=True, help="The chapter of the quiz.")
    parser.add_argument("--quiz-type", type=str, required=True, help="The type of the quiz.")
    parser.add_argument("--python-level", type=str, required=True, help="The Python level of the user.")
    args = parser.parse_args()

    logger.info(f"Chapter: {args.chapter}")
    logger.info(f"Quiz type: {args.quiz_type}")
    logger.info(f"Python level: {args.python_level}")
    logger.success("Command line arguments parsed successfully.\n")

    return args.chapter, args.quiz_type, args.python_level


def get_chapter_content(documents: List[Document], chapter_name: str) -> str:
    """Get the content of a chapter.

    Parameters
    ----------
    documents : List[Document]
        The list of documents.
    chapter : str
        The chapter of the quiz.

    Returns
    -------
    chapter_content : str
        The content of the chapter.
    """
    logger.info("Getting the content of the chapter")
    chapter_content = ""
    # Get the source path of the chapter
    chapter_path = PROCESSED_DATA_PATH + "/" + chapter_name + ".md"
    logger.info(f"Chapter path: {chapter_path}")
    # Get the content of the chapter
    for doc in documents:
        if doc.metadata["source"] == chapter_path:
            chapter_content = doc.page_content
            logger.info(f"Chapter content: {chapter_content[:100]}...")
            logger.info(f"Nb tokens in the chapter content: {calculate_nb_tokens(chapter_content)}")
            logger.success("Chapter content retrieved successfully.\n")
            break
    
    if chapter_content == "":
        logger.error(f"The chapter {chapter_name} not found in the documents.")
        exit()

    # Remove the /n characters from the chapter content
    chapter_content = chapter_content.replace("\n", " ")
    
    return chapter_content


def extract_json_from_string(text: str) -> str:
    """Extract the JSON content from a string.

    Parameters
    ----------
    text : str
        The text containing the JSON content.
    
    Returns
    -------
    json_content : str
        The JSON content.
    """
    logger.info("Extracting the JSON content")
    logger.info(f"Quiz : {text}")

    # Define the delimiters
    start_delimiter = "```json\n"
    end_delimiter = "```"
    
    # Find the start and end delimiters
    start_index = text.find(start_delimiter)
    logger.info(f"Start index: {start_index}")
    if start_index == -1:
        logger.error("Start delimiter not found.")
        return None
    start_index += len(start_delimiter)
    end_index = text.rfind(end_delimiter)
    logger.info(f"End index: {end_index}")
    if end_index == -1:
        logger.error("End delimiter not found.")
        return None
    
    # Extract the JSON content
    json_content = text[start_index:end_index]

    logger.success("JSON content extracted successfully.\n")

    return json_content.strip()


def create_quiz_json(chapter: str, quiz_type: str, level_python: str, chapter_content: str) -> str:
    """Generate a quiz.

    Parameters
    ----------
    chapter : str
        The chapter of the quiz.
    quiz_type : str
        The type of the quiz.
    level_python : str
        The Python level of the user.
    chapter_content : str
        The content of the chapter.

    Returns
    -------
    quiz_json : str
        The generated quiz in JSON format.    
    """
    logger.info("Generating a quiz")

    # Define the model
    chat_model = ChatOpenAI(model=OPENAI_MODEL_NAME)
    # Define the prompt template
    answer_prompt = ChatPromptTemplate.from_template(GENERATE_QUIZ_PROMPT)
    
    # Input data for the prompt
    input_data = {
        "quiz_type": quiz_type,
        "chapter": chapter,
        "level": level_python,
        "chapter_content": chapter_content
    }
    
    # Fill the prompt with the input data
    filled_prompt = answer_prompt.format(**input_data)
    logger.info(f"Filled prompt: {filled_prompt}")
    nb_tokens_prompt = calculate_nb_tokens(filled_prompt)
    logger.info(f"Number of tokens in the prompt: {nb_tokens_prompt}\n")
    
    # Define the chained prompt
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    
    # Generate the answer
    quiz = answer_chain.invoke(input_data)

    # To be sure that the JSON is well formatted
    quiz_json = extract_json_from_string(quiz)

    logger.info(f"Quiz in JSON format: {quiz_json}")
    logger.success("Quiz generated successfully.\n")

    return quiz_json




# MAIN PROGRAM
if __name__ == "__main__":
    # Get the command line arguments
    chapter, quiz_type, level_python = get_args()

    # Load the documents
    documents = load_documents(PROCESSED_DATA_PATH)

    # Get the file names
    file_names = get_file_names(documents)

    # Verify if the chapter exists
    if chapter not in file_names:
        logger.error(f"The chapter {chapter} does not exist.")
        exit()

    # Get the chapter content
    chapter_content = get_chapter_content(documents, chapter)

    # Generate a quiz by LLM
    create_quiz_json(chapter, quiz_type, level_python, chapter_content)
    


