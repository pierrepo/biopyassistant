"""Gradio application to discuss with the course.

Usage:
======
    gradio src/gradio_app_chat.py
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import random

from loguru import logger
import gradio as gr


# MODULES IMPORT
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


# CONSTANTS
QUERY_EXAMPLES = [
    ["Quelle est la différence entre une liste et un set ?"],
    ["Comment faire une boucle en Python ?"],
    ["Comment afficher un float avec 2 chiffres avec la virgule ?"],
]
VECTOR_DB = load_database(CHROMA_PATH)[0]


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
    context = search_similarity_in_database(vector_db=VECTOR_DB, user_query=message)
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


def create_tab_chatbot():
    """Create the interface to discuss with the course."""
    with gr.Blocks(
    theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"),
    title="BioPyAssistant"
    ) as discuss_course_tab:
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
            avatar_images=(("data/img/user_avatar.png"), "data/img/chatbot_avatar.png"),
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
    
    return discuss_course_tab



# MAIN PROGRAM
if __name__ == "__main__":
    # Create the interface to discuss with the course
    discuss_course_tab = create_tab_chatbot()

    # Launch the Gradio interface
    discuss_course_tab.launch(
        server_name="0.0.0.0",  # to make the app accessible from other devices
        inbrowser=True,  # to automatically opens a new tab
    )

