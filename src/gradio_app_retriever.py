"""Gradio app for the model.

Usage:
======
    python src/gradio_app_retriever.py

"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRAIRIES IMPORT
import os
import random

import gradio as gr
from loguru import logger


# MODULES IMPORT
from query_chatbot_by_retriever import load_database, search_similarity_in_database, get_metadata, format_chat_history, calculate_nb_tokens, contextualize_question,  generate_answer, add_metadata_to_answer, MSGS_QUERY_NOT_RELATED


# CONSTANTS
VECTOR_DB_PATH = "chroma_db"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
FLAVICON_RELATIVE_PATH = 'data/logo_round.ico'
QUERY_EXAMPLES= [
    ["C'est quoi la différence entre une liste et un set ?"],
    ["Comment on fait une boucle for en Python ?"], 
    ["Qu'est-ce que la récursivité ?"],
]


# FUNCTIONS
def respond(message: str, chat_history: list, python_level: str) -> str:
    """Respond to the user question.
    
    Parameters
    ----------
    message : str
        The user question.
    chat_history : list
        The chat history.
    python_level : str
        The Python level of the user.
    
    Returns
    -------
    str
        The response to the user question.
    """
    logger.info("Responding to the user question.")

    # Format the chat history for the model
    formatted_chat_history = format_chat_history(chat_history, len_history=10)
    # Contextualize the user question with the chat history
    query_contextualized = contextualize_question(user_query=message, chat_history_formatted=formatted_chat_history, model_name=OPENAI_MODEL_NAME)
    # Search for relevant documents in the database
    context = search_similarity_in_database(vector_db=vector_db, user_query=query_contextualized)
    # If no relevant document was found
    if context == []:
        logger.info("No relevant documents found in the database.")
        logger.success("Returning an automatic response without calling the model.")
        # random response betweet responses in MSGS_QUERY_NOT_RELATED
        response = random.choice(MSGS_QUERY_NOT_RELATED)
        return response
    else :
        # Get the metadata of the relevant documents
        metadata = get_metadata(context)
        # Generate the answer
        answer = generate_answer(query_contextualized=query_contextualized, relevant_chunks=context, model_name=OPENAI_MODEL_NAME, python_level=python_level)
        # Add metadata to the answer
        final_answer = add_metadata_to_answer(answer, metadatas=metadata, iu=True)

        logger.info(f"Response generated: {final_answer}")
        logger.success("Response generated successfully.\n")
                    
        return final_answer


def vote(data: gr.LikeData) -> None: # TODO: Save the votes in a file to do some statistics
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


def display_python_level(data: gr.LikeData):
    """Display in the logs the Python level selected by the user.
    
    Parameters
    ----------
    data : gr.LikeData
        The data containing the Python level information.
    """
    logger.info(f"Python level selected: {data.value}\n")


def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"), title="BioPyAssistant") as demo:
        # Add a title
        gr.HTML("""<h1 text-align="center" style="font-size: 3em;"><center> 🐍 BioPyAssistant 🐍 </center></h1>""")
        
        # Add a description
        with gr.Accordion(label="Description du projet :", open=True):
            gr.HTML("""<p text-align="center" style="font-size: 1em;">
                    Bienvenue sur BioPyAssistant, ton assistant Python pour apprendre à coder en Python. Pose-moi une question sur le cours et je te répondrai !
                    Tu peux aussi me poser des questions sur un chapitre spécifique du cours ou faire un QCM pour tester tes connaissances.
            </p>""")
            # Add the github link
            gr.HTML("""<p text-align="center" style="font-size: 1em;">Pour plus d'informations sur le projet, consulte notre <a href="https://github.com/pierrepo/biopyassistant" target="_blank">dépôt GitHub</a>.</p>""")
    
        # Add a section for asking a question to the chatbot about the course
        with gr.Tab("Chatbot"):
            # Define the query textbox 
            msg = gr.Textbox(placeholder="Pose moi une question sur le cours !", render=False, show_label=False, min_width=200)
            # Define the button for python level
            python_level = gr.Radio(["débutant", "intermédiaire", "avancé"], label="Choisis ton niveau en Python:", value="intermédiaire", render=False)
            # Define the chatbot
            bot = gr.Chatbot(
                value  = [["Hey, j'ai besoin d'aide en Python !", "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?"]], 
                elem_id="chatbot",
                bubble_full_width=False,
                height=300,
                likeable=True,
                show_copy_button=True,
                render=False,
                avatar_images=(
                    ("data/user_avatar.png"), "data/logo_round.webp")
            )
            # Define the chatbot interface
            gr.ChatInterface(
                respond,
                chatbot=bot,
                textbox=msg,
                examples=QUERY_EXAMPLES, 
                cache_examples=False,           
                additional_inputs=[python_level],
                additional_inputs_accordion=gr.Accordion(label="Options avancées", open=False, render=False),
                submit_btn = None,
                retry_btn = "🔄 Réessayer", 
                undo_btn = "↩️ Annuler",
                clear_btn = "🗑️ Supprimer"
            )

            # Display the python level selected
            python_level.select(display_python_level)

            # Adding a like/dislike feature
            bot.like(vote)

    
        # Add a section for asking a qcm about a specific chapter
        with gr.Tab("QCM"):
            pass

    return demo


# MAIN PROGRAM
if __name__ == "__main__":
    # Load the vector database
    vector_db = load_database(VECTOR_DB_PATH)[0]
    
    # Create the the Gradio interface
    demo = create_interface()

    # Get the favicon path
    FLAVICON_PATH = os.path.abspath(FLAVICON_RELATIVE_PATH)
    logger.info(f"Flavicon path: {FLAVICON_PATH}")

    # Launch the Gradio interface
    demo.launch(favicon_path=FLAVICON_PATH,
                 inbrowser=True) # to automatically opens a new tab
    
