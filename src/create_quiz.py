"""Creates the QCM from a chapter of the python course.


Usage:
======
To generate a quiz:
    python src/create_quiz.py --chapter [CHAPTER] --thematic [THEMATIC] --difficulty [DIFFICULTY] --quiz_type [QUIZ_TYPE] --nb_questions [NB_QUESTIONS]
    

Arguments:
==========
    --chapter : str
        The chapter of the quiz.
    --thematic : str
        The thematic of the quiz.
    --difficulty : str
        The difficulty of the quiz.
    --quiz_type : str
        The type of the quiz.
    --nb_questions : int
        The number of questions in the quiz.
   

Example:
========
    python src/create_quiz.py --chapter "Les listes" --thematic "liste en compréhension" --difficulty "Facile" --quiz_type "QCM" --nb_questions 4

This command will create a QCM quiz with 4 questions on the chapter "Les listes" and the thematic "liste en compréhension" with a difficulty "Facile".
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

from loguru import logger
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# MODULE IMPORTS
from query_chatbot_by_retriever import calculate_nb_tokens, OPENAI_MODEL_NAME


# CONSTANTS
GENERATE_QUIZ_PROMPT = """
Tu es un expert en création de quiz sur Python destiné à des étudiants.
Créer un quiz de type {quiz_type} avec {nb_questions} questions sur le chapitre suivant : {chapter} et plus spécifiquement sur cette thématique : {thematic} (si renseigné)
avec un niveau de difficulté {level}.
Tu dois créer {nb_questions} questions variées et pertinentes pour tester les connaissances des étudiants.
Respecte le niveau de difficulté demandé et la thématique si renseignée.
Le format du quiz pourrait être l'un des suivants, respecte le format des exemples données :

- QCM (Questionnaire à Choix Multiples)
    *Exemple*
    1. Quelle est la différence entre une liste et un set ?
        a. Une liste est ordonnée, un set non
        b. Une liste est mutable, un set non
        c. Une liste est indexée, un set non
        d. Une liste est hétérogène, un set non
    
- Vrai ou Faux
    *Exemple*
    1. Python est un langage compilé.
    2. Python est un langage interprété.

- Questions ouvertes
    *Exemple*
    1. Quelle est la différence entre une liste et un set ?

- Exercice de code
    *Exemple*
    1. Ecris une fonction qui prend en paramètre une liste et qui retourne la somme des éléments de cette liste.

- Trouver l'erreur dans le code
    *Exemple*
    1. Quelle est l'erreur dans le code suivant ?
    ```python
    def somme(a, b):
        return a + b
    print(somme(2, 3))
    ```
    a. Il n'y a pas d'erreur
    b. La fonction somme ne retourne pas le bon résultat

- Trouver la sortie du code
    *Exemple*
    1. Quelle est la sortie du code suivant ?
    ```python
    def somme(a, b):
        return a + b
    print(somme(2, 3))
    ```
    a. 5
    b. 6
    c. 7
    d. 8

- Compléter le code
    Les trous sont représentés par des "_" dans le code.
    Plus le niveau de difficulté est élevé, plus le code est complexe et plus il y a de trous à compléter.
    *Exemple*
    1. Complète le code suivant pour qu'il affiche "Hello World"
    ```python
    print("Hello _")
    ```
    a. World
    b. Python
    c. World!
    d. Python!

Respecte le format du quiz demandé.
"""


# FUNCTIONS
def get_args() -> Tuple[str, str, str, str, int]:
    """Parse the command line arguments.

    Returns
    =======
    chapter : str
        The chapter of the quiz.
    thematic : str
        The thematic of the quiz.
    difficulty : str
        The difficulty of the quiz.
    quiz_type : str
        The type of the quiz.
    nb_questions : int
        The number of questions in the quiz.
    """
    logger.info("Parsing command line arguments")

    parser = argparse.ArgumentParser() # Create a parser object
    # Add arguments to the parser
    parser.add_argument("--chapter", type=str, required=True)
    parser.add_argument("--thematic", type=str, default="")
    parser.add_argument("--difficulty", type=str, required=True)
    parser.add_argument("--quiz_type", type=str, required=True)
    parser.add_argument("--nb_questions", type=int, required=True)
    args = parser.parse_args() # Parse the command line arguments

    logger.info("Command line arguments parsed successfully.\n")

    return args.chapter, args.thematic, args.difficulty, args.quiz_type, args.nb_questions


def create_quiz(chapter: str, thematic: str, difficulty: str, quiz_type: str, nb_questions:int) -> str:
    """Generate an quiz.

    Parameters
    ----------
    chapter : str
        The chapter of the quiz.
    thematic : str
        The thematic of the quiz.
    difficulty : str
        The difficulty of the quiz.
    quiz_type : str
        The type of the quiz.
    nb_questions : int
        The number of questions in the quiz.    

    Returns
    -------
    quiz : str
        The generated quiz.    
    """
    logger.info("Generating a quiz")

    # Define the model
    chat_model = ChatOpenAI(model=OPENAI_MODEL_NAME)
    # Define the prompt template
    answer_prompt = ChatPromptTemplate.from_template(GENERATE_QUIZ_PROMPT)
    # Define the chained prompt
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    # Input data for the prompt
    input_data = { "quiz_type": quiz_type, "chapter": chapter, "nb_questions": nb_questions, 'level': difficulty, 'thematic': thematic}
    filled_prompt = answer_prompt.format(**input_data)
    logger.info(f"Filled prompt: {filled_prompt}")
    nb_tokens_prompt = calculate_nb_tokens(filled_prompt)
    logger.info(f"Number of tokens in the prompt: {nb_tokens_prompt}\n")
    # Generate the answer
    quiz = answer_chain.invoke(input_data)
    logger.info(f"Answer: {quiz}")
    logger.success("Answer generated successfully.\n")

    return quiz



# MAIN PROGRAM
if __name__ == "__main__":
    # Get the command line arguments
    chapter, thematic, difficulty, quiz_type, nb_questions = get_args()
    print(f"The {quiz_type} quiz will have {nb_questions} questions on the chapter {chapter} and the thematic {thematic} with a difficulty {difficulty}.\n")

    # Generate a quiz by LLM
    quiz = generate_quiz(chapter, thematic, difficulty, quiz_type, nb_questions)
    
    print(quiz)


