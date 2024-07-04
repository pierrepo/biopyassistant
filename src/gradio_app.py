"""Gradio app to discuss with the course and test your knowledge with a quiz.

Usage:
======
    gradio src/gradio_app.py

"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRAIRIES IMPORT
import os
import json
import random
from typing import Tuple

import re
import gradio as gr
from loguru import logger


# MODULES IMPORT
from create_database import load_documents, get_file_names
from query_chatbot import (
    load_database,
    search_similarity_in_database,
    get_metadata,
    format_chat_history,
    contextualize_question,
    generate_answer,
    add_metadata_to_answer,
    format_relevant_chunks,
    MSGS_QUERY_NOT_RELATED,
    OPENAI_MODEL_NAME,
    CHROMA_PATH,
)
from create_quiz import create_quiz_json, get_chapter_content


# CONSTANTS
FLAVICON_RELATIVE_PATH = "data/img/logo_round.ico"
PROCESSED_DATA_PATH = "data/markdown_processed"
QUERY_EXAMPLES = [
    ["Quelle est la différence entre une liste et un set ?"],
    ["Comment faire une boucle en Python ?"],
    ["Comment afficher un float avec 2 chiffres avec la virgule ?"],
]
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
def respond(message: str, chat_history: list) -> str:
    """Respond to the user question.

    Parameters
    ----------
    message : str
        The user question.
    chat_history : list
        The chat history.

    Returns
    -------
    str
        The response to the user question.
    """
    logger.info("Responding to the user question...")

    # Format the chat history for the model
    formatted_chat_history = format_chat_history(chat_history, len_history=10)
    # Contextualize the user question with the chat history
    chat_context = contextualize_question(chat_history_formatted=formatted_chat_history)
    # Search for relevant documents in the database
    context = search_similarity_in_database(vector_db=vector_db, user_query=message)
    # If no relevant document was found
    if context == []:
        logger.info("No relevant documents found in the database.")
        logger.success("Returning an automatic response without calling the model.")
        # random response betweet responses in MSGS_QUERY_NOT_RELATED
        response = random.choice(MSGS_QUERY_NOT_RELATED)
        return response
    else:
        # Format the relevant chunks
        context_formatted = format_relevant_chunks(context)
        # Get the metadata of the relevant documents
        metadata = get_metadata(context)
        # Generate the answer
        answer = generate_answer(
            query=message,
            chat_context=chat_context,
            relevant_chunks=context_formatted,
            model_name=OPENAI_MODEL_NAME,
        )
        # Add metadata to the answer
        final_answer = add_metadata_to_answer(answer, metadatas=metadata, iu=True)

        return final_answer


def vote(
    data: gr.LikeData,
) -> None:  # TODO: Save the votes in a file to do some statistics
    """Display in the logs the vote of the user.

    Parameters
    ----------
    data : gr.LikeData
        The data containing the vote information.
    """
    if data.liked:
        logger.info(f"You upvoted this response: {data.value}\n")
    else:
        logger.info(f"You downvoted this response: {data.value}\n")


def get_user_input(data: gr.SelectData) -> str:
    """Get the user input.

    Parameters
    ----------
    data : gr.SelectData
        The data containing the user input.

    Returns
    -------
    str
        The user input.
    """
    logger.info(f"{data.target.label} {data.value}\n")

    return data.value


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


def generate_quiz(
    chapter: str, quiz_type: str, python_level: str
) -> Tuple[str, dict, str, dict]:
    """Generate a quiz to ask the user.

    Parameters
    ----------
    chapter : str
        The chapter of the question.
    quiz_type : str
        The type of the quiz.
    python_level : str
        The Python level of the user.

    Returns
    -------
    Tuple[str, dict, str, dict]
        The question, the answers, the correct answer and the explanation.

    """
    logger.info("Generating a quiz.")

    # Display the quiz parameters
    logger.info(f"Chapter: {chapter}")
    logger.info(f"Quiz type: {quiz_type}")
    logger.info(f"Python level: {python_level}")

    # Get the file names of the chapters
    chapter_file_names = get_chapters_files_name(chapters)
    chapter_file_name = chapter_file_names[chapter]

    # Get the chapter content
    chapter_content = get_chapter_content(
        documents=chapters, chapter_name=chapter_file_name
    )

    # Create the quiz
    quiz_json = create_quiz_json(chapter, quiz_type, python_level, chapter_content)

    return quiz_json


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


def create_tab_discuss_course():
    """Create the interface to discuss with the course."""
    # Define the query textbox
    msg = gr.Textbox(
        placeholder="Pose moi une question sur le cours !",
        render=False,
        show_label=False,
        min_width=200,
    )
    # Define the button for python level
    python_level = gr.Radio(
        ["débutant", "intermédiaire", "avancé"],
        label="Choisis ton niveau en Python:",
        value="intermédiaire",
        render=False,
    )
    # Define the chatbot
    bot = gr.Chatbot(
        value=[
            [
                "Hey, j'ai besoin d'aide en Python !",
                "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?",
            ]
        ],
        bubble_full_width=False,
        height=600,
        likeable=True,
        show_copy_button=True,
        render=False,
        avatar_images=(("data/img/user_avatar.png"), "data/img/logo_round.webp"),
    )
    # Define the chatbot interface
    gr.ChatInterface(
        respond,
        chatbot=bot,
        textbox=msg,
        examples=QUERY_EXAMPLES,
        cache_examples=False,
        submit_btn=None,
        retry_btn="🔄 Reposer la question",
        undo_btn="↩️ Annuler la dernière question",
        clear_btn="🗑️ Supprimer la conversation",
    )

    # Display the python level selected
    python_level.select(get_user_input)

    # Adding a like/dislike feature
    bot.like(vote)


def create_tab_quiz():
    """Create the interface to generate a quiz."""
    with gr.Blocks(
    theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"),
    title="BioPyAssistant",
    css=CSS
    ):

        gr.Markdown("*Sélectionnez les paramètres du quiz*")
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


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"),
        title="BioPyAssistant",
        css=CSS
    ) as demo:
        # Add a title
        gr.HTML(
            """<h1 style="font-size: 3em;"><center> 🐍 BioPyAssistant 🐍 </center></h1>"""
        )

        # Add a section for asking a question to the chatbot about the course
        with gr.Tab("Discuter avec le cours"):
            create_tab_discuss_course()

        # Add a section for asking a qcm about a specific chapter
        with gr.Tab("Se tester"):
            create_tab_quiz()

    return demo


# MAIN PROGRAM
if __name__ == "__main__":
    # Load the vector database
    vector_db = load_database(CHROMA_PATH)[0]

    # Get the content of each chapter
    chapters = load_documents(PROCESSED_DATA_PATH)

    # Get the chapter names
    chapter_names = get_chapters_files_name(chapters)

    # Create the the Gradio interface
    demo = create_interface()

    # Get the favicon path
    FLAVICON_PATH = os.path.abspath(FLAVICON_RELATIVE_PATH)
    logger.info(f"Flavicon path: {FLAVICON_PATH}")

    # Launch the Gradio interface
    demo.launch(
        favicon_path=FLAVICON_PATH,  # to add a favicon
        server_name="0.0.0.0",  # to make the app accessible from other devices
        inbrowser=True,  # to automatically opens a new tab
    )