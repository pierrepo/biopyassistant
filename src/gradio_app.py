"""Gradio app for the model.

Usage:
======
    python src/gradio_app.py

"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRAIRIES IMPORT
import os
import csv

import gradio as gr
from loguru import logger


# MODULES IMPORT
from query_chatbot import load_database, search_similarity_in_database, get_metadata, generate_prompt, predict_response, adding_metadatas_to_response, MSGS_QUERY_NOT_RELATED


# CONSTANTS
VOTES_DATA = []
VECTOR_DB_PATH = "chroma_db"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"
FLAVICON_RELATIVE_PATH = 'data/logo_round.ico'
QUERY_EXAMPLES= [
    ["C'est quoi la différence entre une liste et un set ?"],
    ["Comment on fait une boucle for en Python ?"], 
    ["Qu'est-ce que la récursivité ?"],
]



# FUNCTIONS
def generate_response_with_gradio(user_query, python_level):
    """Generate a response to the user question."""

    # Search for relevant documents in the database
    relevant_chunks = search_similarity_in_database(vector_db, user_query)

    # If no relevant document was found, return the response from the model
    if relevant_chunks == []:
        logger.info("No relevant documents found in the database.")
        logger.success("Returning an automatic response without calling the model.")
        return MSGS_QUERY_NOT_RELATED[0]

    else: # If relevant documents were found
        # Generate a prompt for the AI model
        prompt = generate_prompt(relevant_chunks, user_query, python_level)[0]

        # Predict the response using the AI model and get the number of tokens in the response
        response_from_model = predict_response(prompt, OPENAI_MODEL_NAME)[0]
    
        # Get the metadata of the most similar document
        metadatas = get_metadata(relevant_chunks)
        
        # Add metadata to the response
        response_with_metadata = adding_metadatas_to_response(response_from_model, metadatas, iu=True)

        return response_with_metadata


def respond(message, chat_history, python_level):
    bot_message = generate_response_with_gradio(message, python_level)
    chat_history.append((message, bot_message))
    return "", chat_history


def undo(msg, chat_history):
    """Undo the last message."""
    pass


def retry(chat_history, python_level): # DOES NOT WORK
    """Retry the last message."""
    logger.info("Retrying the last message.")
    # Check if there is a chat history
    if chat_history:
        # Get the last message
        last_msg = chat_history[-1][0]
        logger.info(f"Last message: {last_msg}")
        # Respond to the last message
        response = respond(last_msg, chat_history, python_level)
    return response
    

def vote(data: gr.LikeData) -> None: # TODO: Save the votes in a file to do some statistics
    """Vote for a response to a user query."""
    if data.liked:
        logger.info(f"You upvoted this response: {data.value}\n")
    else:
        logger.info(f"You downvoted this response: {data.value}\n")


def display_python_level(data: gr.LikeData):
    """Display the selected Python level."""
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
            # Define the chatbot
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                height=500,
                likeable=True,
                show_copy_button=True,
                avatar_images=(
                    ("data/user_avatar.png"), "data/logo_round.webp"),
                value  = [["Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?", "Coucou"]], 
            )
            # Define the query textbox 
            msg = gr.Textbox(placeholder="Pose moi une question sur le cours !", render=False, show_label=False)

            # Define the buttons for retrying, undoing and clearing the chat history
            with gr.Row():
                retry_btn = gr.Button("🔄 Réessayer", size='sm' )
                undo_btn = gr.Button("↩️ Annuler", size='sm' )
                clear = gr.ClearButton([msg, chatbot], value= "🗑️ Supprimer", size='sm' )
            msg.render() # render the message box after the buttons

            # Define example questions 
            query_exemple = gr.Examples(examples=QUERY_EXAMPLES, inputs=[msg], fn=respond, label= "Exemples de questions:")

            # Define the accordion for advanced options
            with gr.Accordion("Options avancées", open=False):
                # Define the button for python level
                python_level = gr.Radio(["débutant", "intermédiaire", "avancé"], label="Choisis ton niveau en Python:", value="intermédiaire")
            # Display the python level selected
            python_level.select(display_python_level)

            # Define the button for submitting the message
            msg.submit(respond, inputs=[msg, chatbot, python_level], outputs=[msg, chatbot])

            # Adding a like/dislike feature
            chatbot.like(vote)

            # Define the undo button
            undo_btn.click(undo, inputs=[msg, chatbot])

            # Define the retry button
            retry_btn.click(retry, inputs=[chatbot, python_level])

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
    
