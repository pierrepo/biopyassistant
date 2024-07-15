"""Creates Quiz from a chapter of the python course in JSON format using a LLM. 


Usage:
======
    python src/create_quiz.py --chapter [CHAPTER] --quiz-type [QUIZ_TYPE] --python-level [PYTHON_LEVEL]

Arguments:
==========
    --chapter : str
        The name of the chapter of the quiz. It must be one of the chapters of the Python course contained in the data/markdown_processed folder.
        Attention: The chapter name must be written in French and with the first letter of each word capitalized.
    --quiz-type : str
        The type of the quiz (QCM or Vrai/Faux).
    --python-level : str
        The Python level of the user (Débutant or Avancé).

Example:
========
    python src/create_quiz.py --chapter Listes --quiz-type QCM --python-level Débutant

This command will create a quiz of 5 QCM on the chapter "Listes" for a beginner level user.
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import os
import json
import argparse
from typing import Tuple
from datetime import datetime


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

QUIZ_JSON_PATH = "data/quiz.json"

GENERATE_QUIZ_PROMPT = """
Tu es un expert en création de quiz sur Python destiné à des étudiants.
Créer un quiz de type {quiz_type} avec 5 questions sur le chapitre suivant : {chapter} pour un étudiant de niveau {level}.
La question doit être au format markdown et doit porter sur le contenu du chapitre fourni et doit être accompagnée de réponses possibles et d'une explication détaillée pour chaque réponse.
N'hésites pas à varier les questions (ex: questions basés sur le cours, trouver l'erreur dans le code, trouver la sortie du code, compléter le code, etc.).

Le format de sortie doit seulement etre un JSON comportant les 5 questions de quiz avec les clés suivantes :
- type (str): le type du quiz (QCM, Vrai ou Faux)
- chapter (str): le chapitre du quiz
- python_level (str): le niveau Python de l'étudiant
- question (str): la question posée
- answers (dict): les réponses possibles numérotées par ordre alphabétique (A, B, C, D)
- correct_answer (str): la réponse correcte avec la lettre correspondante
- explanation (str): l'explication détaillée pour chaque réponse possible

Je veux dans explanation, une explication détaillée pour chaque réponse possible de manière à ce que l'étudiant comprenne pourquoi chaque réponse est correcte ou incorrecte.
Le ton doit être pédagogique et bienveillant.

Voici un exemple de format de sortie attendue :
{{
    "questions": [
        {{
            "chapter": "Variables",
            "python_level": "Débutant",
            "type": "QCM",
            "question": "Quelle est la sortie du code suivant : \\n```python\\nx = 5\\ny = 10\\nprint(x + y)\\n```",
            "answers": {{
                "A": "10",
                "B": "15",
                "C": "5",
                "D": "50"
            }},
            "correct_answer": "B",
            "explanation": "La réponse correcte est B car x vaut 5 et y vaut 10, donc x + y vaut 15. Les autres réponses sont incorrectes car elles ne correspondent pas à la somme de x et y."
        }},
        {{
            "chapter": "Variables",
            "python_level": "Débutant",
            "type": "QCM",
            "question": "Quelle structure conditionnelle est utilisée pour exécuter un bloc de code si une condition est vraie ?",
            "answers": {{
                "A": "for",
                "B": "while",
                "C": "if",
                "D": "def"
            }},
            "correct_answer": "C",
            "explanation": "La réponse correcte est C car la structure conditionnelle if est utilisée pour exécuter un bloc de code si une condition est vraie. Les réponses A ("for") et B ("while") sont utilisées pour les boucles afin de répéter un bloc de code plusieurs fois et la réponse D ("def") est utilisée pour définir une fonction."
        }}
    ]
}}

Voici le contenu du chapitre sur lequel tu dois créer le quiz :
{chapter_content}
"""


# FUNCTIONS
def get_chapters_file_names(documents: list) -> dict:
    """Get the names of the chapters.

    Parameters
    ----------
    documents : list
        The list of documents.

    Returns
    -------
    chapters : dict
        The chapter names as keys and the chapter file names as values.
    """
    logger.info("Getting the names of the chapters...")
    chapters = {}

    # Get the names of the chapters
    file_names = get_file_names(documents)
    
    for file_name in file_names:
        # Define the pattern to match the chapter name
        match_chapter = re.match(r"\d+_(.*)$", file_name)
        if match_chapter:
            # Get the chapter name and format it
            chapter_name = match_chapter.group(1).replace("_", " ").capitalize()
            chapters[chapter_name] = file_name
    
    logger.success("Chapter names with their content files retrieved successfully.\n")

    return chapters


def get_args(chapter_names: list[str]) -> Tuple[str, str, str]:
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

    # Checks
    if args.chapter not in chapter_names:
        logger.error(f"The chapter {args.chapter} is not valid. Please choose one of the following chapters: {', '.join(chapter_names)}.")
        exit()
    if args.quiz_type not in ["QCM", "Vrai/Faux"]:
        logger.error(f"The quiz type {args.quiz_type} is not valid. Please choose between 'QCM' and 'Vrai/Faux'.")
        exit()
    if args.python_level not in ["Débutant", "Avancé"]:
        logger.error(f"The Python level {args.python_level} is not valid. Please choose between 'Débutant' and 'Avancé'.")
        exit()
    logger.success("Command line arguments parsed successfully.\n")

    return args.chapter, args.quiz_type, args.python_level


def get_chapter_content(chapters: dict[str, str], chapter_name: str) -> str:
    """Get the content of a chapter.

    Parameters
    ----------
    chapters : dict[str, str]
        Dictionary containing the chapters names and their content.
    chapter : str
        The chapter of the quiz.

    Returns
    -------
    chapter_content : str
        The content of the chapter.
    """
    logger.info(f"Getting the content of the chapter: {chapter_name}")

    # Get the content of the chapter
    chapter_file_name = chapters[chapter_name] + ".md"
    chapter_file_path = os.path.join(PROCESSED_DATA_PATH, chapter_file_name)

    with open(chapter_file_path, "r") as file:
        chapter_content = file.read()
    
    # Remove the /n characters from the chapter content
    chapter_content = re.sub(r"\n", "", chapter_content)
    logger.info(f"Chapter content: {chapter_content[:100]}...")

    logger.success("Chapter content retrieved successfully.\n")
    
    return chapter_content


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
    logger.info(f"Quiz: {quiz}")
    logger.success("Quiz generated successfully.\n")

    return quiz


def extract_json_from_string(quiz: str) -> str:
    """Extract JSON part from a string that contains JSON."""
    try:
        start_index = quiz.index('{')
        end_index = quiz.rindex('}') + 1
        return quiz[start_index:end_index]
    except ValueError:
        logger.error("No valid JSON found in the input string.")
        raise


def validate_quiz_json(quiz: str) -> str:
    """Validate the quiz JSON.

    Parameters
    ----------
    quiz : str
        The quiz generated by the LLM.

    Returns
    -------
    quiz_json : str
        The validated quiz in JSON format.
    """
    logger.info("Validating the quiz JSON...")

    # Extract the JSON content from the quiz
    quiz_json = extract_json_from_string(quiz)

    # Validate the JSON content
    try:
        quiz_data = json.loads(quiz_json)
    except json.JSONDecodeError as e:
        logger.error(f"The JSON is not well-formed : {e}")
        exit()

    # Validate each question
    for question in quiz_data["questions"]:
        question_type = question.get("type")
        answers = question.get("answers", {})
        correct_answer = question.get("correct_answer")

        # Validate the number of answers
        if question_type == "QCM" and len(answers) != 4:
            logger.error(f"Invalid number of answers for 'QCM' question: {question['question']}")
        elif question_type == "Vrai ou Faux" and len(answers) != 2:
            logger.error(f"Invalid number of answers for 'Vrai ou Faux' question: {question['question']}")

        # Validate the correct answer is one of the possible answers
        if correct_answer not in answers.keys():
            logger.error(f"Invalid correct answer for question: {question['question']}")
        
    logger.success("Quiz JSON validated successfully.\n")

    return quiz_json


def save_to_json(quiz_json: str, file_path: str) -> None:
    """Save the quiz in a JSON file.

    Parameters
    ----------
    quiz_json : str
        The generated quiz in JSON format.
    file_path : str
        The path of the JSON file.
    """
    logger.info(f"Saving the quiz in a JSON file: {file_path}")

    # Convert json into a dictionary
    new_quiz_data = json.loads(quiz_json)
    
    # Add the last modified date to the new quiz data
    last_modified = datetime.now().isoformat()
    new_quiz_data['last_modified'] = last_modified

    # Load existing quiz data if the file exists
    if os.path.exists(file_path):
        logger.info("The file already exists. Merging the new quiz data with the existing data.")
        with open(file_path, "r", encoding="utf-8") as file:
            existing_quiz_data = json.load(file)
        # Merge the new quiz data with the existing data
        existing_quiz_data["questions"] = existing_quiz_data.get("questions", []) + new_quiz_data.get("questions", [])
        # Update the last modified date
        existing_quiz_data['last_modified'] = last_modified
        # Convert the merged data back to JSON
        merged_quiz_json = json.dumps(existing_quiz_data, ensure_ascii=False, indent=4, sort_keys=False)
    else:
        logger.info("The file does not exist. Saving the new quiz data.")
        # If the file does not exist, use the new quiz data
        merged_quiz_json = json.dumps(new_quiz_data, ensure_ascii=False, indent=4, sort_keys=False)

    # Save the merged quiz data to the file in UTF-8 encoding
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(merged_quiz_json)
    
    logger.success("Quiz saved successfully.\n")


# MAIN PROGRAM
if __name__ == "__main__":
    # Get chapters information
    CHAPTERS = load_documents(PROCESSED_DATA_PATH)
    CHAPTERS_DIC = get_chapters_file_names(CHAPTERS)
    CHAPTER_NAMES = list(CHAPTERS_DIC.keys())

    # Get the command line arguments
    chapter, quiz_type, level_python = get_args(CHAPTER_NAMES)

    # Get the chapter content
    chapter_content = get_chapter_content(CHAPTERS_DIC, chapter)

    # Generate a quiz by LLM
    quiz = create_quiz_json(chapter, quiz_type, level_python, chapter_content)

    # Validate the quiz JSON
    quiz_json = validate_quiz_json(quiz)

    # Save the quiz in a JSON file
    save_to_json(quiz_json, QUIZ_JSON_PATH)
    
