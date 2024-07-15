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
import os
import json
import random

from loguru import logger
import gradio as gr


# MODULES IMPORT
from create_database import load_documents
from create_quiz import get_chapters_file_names


# CONSTANTS
PROCESSED_DATA_PATH = "data/markdown_processed"
QUIZ_TYPES = [
    "QCM",
    "Vrai/Faux"
]

PATH_QUIZ_JSON = "data/quiz.json"

CSS = """
#correct-answer
    {background-color: #a3e288}
#wrong-answer
    {background-color: #dd553e}
"""


# FUNCTIONS
def load_quiz(path_quiz_json: str) -> dict:
    """Load the json file that contains the quizzes.

    Parameters
    ----------
    path_quiz_json : str
        The path to the json file.
    
    Returns
    -------
    dict
        The quizzes.
    """
    logger.info("Loading the quiz json file...")
    try:
        full_path = os.path.join(os.getcwd(), path_quiz_json)
        with open(full_path, 'r') as file:
            quiz_json = json.load(file)
        
        logger.success("The json file has been loaded.\n")
        return quiz_json
    except FileNotFoundError:
        logger.error("The json file does not exist.")
        return None
    except json.JSONDecodeError:
        logger.error("Error decoding the json file.")
        return None
    

def make_quiz(chapter_choice: str, level_choice: str, quiz_type_choice: str):
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
    logger.info("Retrieving a quiz...")

    # Display the quiz parameters
    logger.info(f"Chapter: {chapter_choice}")
    logger.info(f"Level: {level_choice}")
    logger.info(f"Quiz type: {quiz_type_choice}")

    # Load json that contains the quizzes
    quiz_json = load_quiz(PATH_QUIZ_JSON)

    # Filter questions based on user choices
    filtered_questions = []
    for question in quiz_json["questions"]:
        if (question["chapter"] == chapter_choice and
            question["difficulty"] == level_choice and
            question["type"] == quiz_type_choice):
            filtered_questions.append(question)

    # Test if there is a quiz for the given parameters
    if filtered_questions == []:
        logger.error("No quiz found for the given parameters.")
        return (
            gr.Markdown("Désolé, aucun quiz n'a été trouvé pour les paramètres donnés."),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

    # Get the proper quiz
    QUIZ = random.choice(filtered_questions)

    if quiz_type_choice == "Vrai ou Faux":
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
        return (
            gr.Markdown(QUIZ["question"], visible=True),
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
        CHAPTERS = load_documents(PROCESSED_DATA_PATH)
        CHAPTERS_DIC = get_chapters_file_names(CHAPTERS)
        CHAPTER_NAMES = list(CHAPTERS_DIC.keys())

        with gr.Row():
            chapter_choice = gr.Dropdown(
                choices=CHAPTER_NAMES,
                value=CHAPTER_NAMES[0],
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