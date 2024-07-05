""" Gradio application to generate a quiz.

Usage:
======
    gradio src/gradio_app_quiz.py
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import json
import re
from typing import Tuple

from loguru import logger
import gradio as gr


# MODULES IMPORT
from create_database import load_documents, get_file_names
from create_quiz import create_quiz_json, get_chapter_content


# CONSTANTS
PROCESSED_DATA_PATH = "data/markdown_processed"
QUIZ_TYPES = [
    "QCM (Questionnaire à Choix Multiples)",
    "Vrai ou Faux",
    "Trouver l'erreur dans le code",
    "Trouver la sortie du code",
    "Compléter le code",
]

QUIZ_JSON = """
{
    "type": "QCM",
    "question": "Question de QCM",
    "answers": {
        "A": "réponse A",
        "B": "réponse B",
        "C": "réponse C",
        "D": "réponse D"
    },
    "correct_answer": "C",
    "explanation": "Parce que C est la bonne réponse"
}
"""

QUIZ_JSON_VRAI_FAUX = """
{
    "type": "Vrai-faux",
    "question": "Question de Vrai ou Faux",
    "answers": {
        "A": "Vrai",
        "B": "Faux"
    },
    "correct_answer": "Vrai",
    "explanation": "Parce que A est la bonne réponse"
}
"""

CSS = """
#correct-answer
    {background-color: #a3e288}
#wrong-answer
    {background-color: #dd553e}
"""


# FUNCTIONS
def get_chapters_files_name(documents: list) -> dict:
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
    
    chapter_names = list(chapters.keys())
    logger.success("Chapter names retrieved successfully.\n")

    return chapter_names


def extract_quiz_elements(json_data: str) -> Tuple[str, dict, str, dict]:
    """
    Extracts elements from the provided JSON data.

    Parameters
    ----------
    json_data : str
        JSON data as a string.

    Returns
    -------
    Tuple[str, dict, str, dict]
        The question, the answers, the correct answer and the explanation.
    """
    logger.info("Extracting quiz elements.")
    # Parse the JSON data
    quiz_data = json.loads(json_data)

    # Extract elements
    quiz_elements = {
        "type": quiz_data.get("type"),
        "question": quiz_data.get("question"),
        "answers": quiz_data.get("answers"),
        "correct_answer": quiz_data.get("correct_answer"),
        "explanation": quiz_data.get("explanation"),
    }

    # Get the elements
    question = quiz_elements["question"]
    answers = quiz_elements["answers"]
    correct_answer = quiz_elements["correct_answer"]
    explanation = quiz_elements["explanation"]

    logger.info(f"Question: {question}")
    logger.info(f"Answers: {answers}")
    logger.info(f"Correct answer: {correct_answer}")
    logger.info(f"Explanation: {explanation}\n")
    logger.success("Quiz elements extracted successfully.\n")

    return question, answers, correct_answer, explanation



def make_quiz(chapter_choice, level_choice, quiz_type_choice):
    """Generate a quiz to ask the user.

    Parameters
    ----------
    chapter_choice : str
        The chapter of the question.
    level_choice : str
        The level of the user.
    quiz_type_choice : str
        The type of the quiz.
    """
    logger.info("Generating a quiz...")

    # Display the quiz parameters
    logger.info(f"Chapter: {chapter_choice}")
    logger.info(f"Level: {level_choice}")
    logger.info(f"Quiz type: {quiz_type_choice}")

    if quiz_type_choice == "Vrai ou Faux":
        # Generate the quiz
        QUIZ = json.loads(QUIZ_JSON_VRAI_FAUX)
        return (
            gr.Markdown("## " + QUIZ["question"], visible=True),
            None,
            None,
            None,
            None,
            gr.Radio(
                label="Sélectionnez la proposition correcte :",
                choices=["Vrai", "Faux"],
                value=None,
                visible=True
            ),
            QUIZ["correct_answer"],
            QUIZ["explanation"],
            gr.Markdown("", visible=False)
        )
    else:
        # Generate the quiz
        QUIZ = json.loads(QUIZ_JSON)
        return (
            gr.Markdown("## Question : " + QUIZ["question"], visible=True),
            gr.Markdown("## Proposition A\n\n" + QUIZ["answers"]["A"], visible=True),
            gr.Markdown("## Proposition B\n\n" + QUIZ["answers"]["B"], visible=True),
            gr.Markdown("## Proposition C\n\n" + QUIZ["answers"]["C"], visible=True),
            gr.Markdown("## Proposition D\n\n" + QUIZ["answers"]["D"], visible=True),
            gr.Radio(
                label="Sélectionnez la proposition correcte :",
                choices=list(QUIZ["answers"].keys()),
                value=None,
                visible=True
            ),
            QUIZ["correct_answer"],
            QUIZ["explanation"],
            gr.Markdown("", visible=False)
        )


def check_answer(quiz_radio, correct_answer, explanation_text):
    if quiz_radio == correct_answer:
        message = (f"## Réponse\n\nBravo ! "
                   f"La réponse correct est {correct_answer} \n\n"
                   f"{explanation_text}")
        css_id = "correct-answer"
    else:
        message = (f"## Réponse\n\n"
                   f"Désolé, ce n'est pas la réponse {quiz_radio}. "
                   f"La bonne réponse est {correct_answer}.\n\n"
                   f"{explanation_text}")
        css_id = "wrong-answer"
    return gr.Radio(visible=False), gr.Markdown(value=message, visible=True, elem_id=css_id)


def create_tab_quiz():
    """Create the interface to generate a quiz."""
    with gr.Blocks(
    theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"),
    title="BioPyAssistant",
    css=CSS
    ) as quiz_tab:

        gr.Markdown("*Sélectionnez les paramètres du quiz*")
        # Get the chapter names
        chapters = load_documents(PROCESSED_DATA_PATH)
        chapter_names = get_chapters_files_name(chapters)

        with gr.Row():
            chapter_choice = gr.Dropdown(
                choices=chapter_names,
                value=chapter_names[0],
                label="Chapitre :")
            level_names = ["Débutant", "Confirmé"]
            level_choice = gr.Dropdown(
                choices=level_names,
                value=level_names[0],
                label="Niveau de difficulté :"
            )
        
            quiz_type_choice = gr.Dropdown(
                choices=QUIZ_TYPES,
                value=QUIZ_TYPES[0],
                label="Type :"
            )
            create_quiz_button = gr.Button("Générer un quiz", size="sm")

        with gr.Blocks():
            question = gr.Markdown(visible=False)
            if quiz_type_choice == "Vrai ou Faux":
                with gr.Row(equal_height=True):
                    answer_1_md = gr.Markdown(label="Réponse 1 :", visible=False)
                    answer_2_md = gr.Markdown(label="Réponse 2 :", visible=False)
            else:
                with gr.Row(equal_height=True):
                    answer_1_md = gr.Markdown(label="Réponse 1 :", visible=False)
                    answer_2_md = gr.Markdown(label="Réponse 2 :", visible=False)
                with gr.Row(equal_height=True):
                    answer_3_md = gr.Markdown(label="Réponse 3 :", visible=False)
                    answer_4_md = gr.Markdown(label="Réponse 4 :", visible=False)
            
            answer_choice = gr.Radio(visible=False)
            explanation_text = gr.Textbox(visible=False)
            correct_answer_text = gr.Textbox(visible=False)
            explanation_md = gr.Markdown(visible=False)


        create_quiz_button.click(
            fn=make_quiz,
            inputs=[chapter_choice, level_choice, quiz_type_choice],
            outputs=[
                question,
                answer_1_md, answer_2_md, answer_3_md, answer_4_md,
                answer_choice, correct_answer_text, explanation_text, explanation_md
            ]
        )

        answer_choice.input(
            fn=check_answer,
            inputs=[answer_choice, correct_answer_text, explanation_text],
            outputs=[answer_choice, explanation_md]
            )
        
    return quiz_tab
        

if __name__ == "__main__":
    # Create the Gradio interface for the quiz
    quiz_tab = create_tab_quiz()

    # Launch the Gradio interface
    quiz_tab.launch(
        server_name="0.0.0.0",  # to make the app accessible from other devices
        inbrowser=True,  # to automatically opens a new tab
    )
