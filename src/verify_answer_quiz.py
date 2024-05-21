"""Verify the answers for a quiz on Python.


Usage:
======
    python src/verify_answer_quiz.py --answer [ANSWER] --quiz [QUIZ] --quiz_type [QUIZ_TYPE]
    

Arguments:
==========
    --answer : str
        The answer of the student.
    --quiz : str
        The quiz to answer.
    --quiz_type : str
        The type of the quiz.
   

Example:
========
    python src/verify_answer_quiz.py --answer "a" --quiz "<C'est quoi Python ?> : <a. Un langage de programmation>, <b. Un animal>, <c. Un objet>, <d. Un fruit>" --quiz_type "QCM"

This command will verify the answer "a" for the quiz "<C'est quoi Python ?> : <a. Un langage de programmation>, <b. Un animal>, <c. Un objet>, <d. Un fruit>", which is a QCM quiz.
It will return if the answer is correct or not and why.
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
VERIFY_ANSWERS_PROMPT = """
Tu es un assistant virtuel d'éducation destiné à vérifier les réponses d'un quiz de type {quiz_type} sur Python.
Voici la/les question(s) du quiz :
{quiz}
Voici la réponse pour chaque question :
{answers}

Pour chaque question, tu dois vérifier si la réponse est correcte ou non et le dire avec un encouragement ou un bravo si la réponse est correcte.
Je veux que tu sois pointilleux, si la réponse est partiellement correcte, dis le.
Tu dois donner un feedback détaillé pour chaque question, en expliquant pourquoi la réponse est correcte ou non.
"""


# FUNCTIONS
def get_args() -> Tuple[str, str, str]:
    """Parse the command line arguments.

    Returns
    -------
    Tuple[str, str, str]
        The answer of the student, the quiz and the quiz type.
    """
    logger.info("Parsing command line arguments")

    parser = argparse.ArgumentParser() # Create a parser object
    # Add arguments to the parser
    parser.add_argument("--answer", type=str, required=True, help="The answer of the student.")
    parser.add_argument("--quiz", type=str, required=True, help="The quiz to answer.")
    parser.add_argument("--quiz_type", type=str, required=True, help="The type of the quiz.")
    args = parser.parse_args() # Parse the command line arguments

    # Add Checks
    if args.quiz_type not in ["QCM", "Vrai-Faux", "Questions ouvertes"]:
        logger.error(f"Quiz type must be 'QCM', 'Vrai-Faux' or 'Question ouverte', not {args.quiz_type}")

    return args.answer, args.quiz, args.quiz_type


def generate_feedback(answer: str, quiz: str, quiz_type: str) -> str:
    """Generate an answer to the student's quiz.

    Parameters
    ----------
    answer : str
        The answer of the student.
    quiz : str
        The quiz to answer.
    quiz_type : str
        The type of the quiz.

    Returns
    -------
    feedback : str
        The feedback generated by the model.
    """
    logger.info("Generating an answer to the user query.")

    # Define the model
    chat_model = ChatOpenAI(model=OPENAI_MODEL_NAME)
    # Define the prompt template
    answer_prompt = ChatPromptTemplate.from_template(VERIFY_ANSWERS_PROMPT)
    # Define the chained prompt
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    # Input data for the prompt
    input_data = { "quiz_type": quiz_type, "quiz": quiz, "answers": answer }
    filled_prompt = answer_prompt.format(**input_data)
    logger.info(f"Filled prompt: {filled_prompt}")
    nb_tokens_prompt = calculate_nb_tokens(filled_prompt)
    logger.info(f"Number of tokens in the prompt: {nb_tokens_prompt}\n")
    # Generate the answer
    feedback = answer_chain.invoke(input_data)
    logger.info(f"Answer: {answer}")
    logger.success("Answer generated successfully.\n")

    return feedback


# MAIN PROGRAM
if __name__ == "__main__":
    # Get the command line arguments
    answer, quiz, quiz_type = get_args()

    # Generate answer from the llm
    feedback = generate_feedback(answer, quiz, quiz_type)

    print(feedback)
