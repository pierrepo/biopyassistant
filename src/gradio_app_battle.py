"""Gradio application to discuss with the course in a battle mode.

The user will ask a question and it will be answered by 2 random different models. The user will have to choose wich answer is the best. 

Usage:
======
    gradio src/gradio_app_battle.py
"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRARY IMPORTS
import random
from typing import Tuple

import gradio as gr
from loguru import logger
from dotenv import load_dotenv


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
    CHROMA_PATH,
)


# CONSTANTS
VECTOR_DB = load_database(CHROMA_PATH)[0]
LLM_MODELS = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it", "open-mistral-7b", "open-mixtral-8x7b", "open-mixtral-8x22b", "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
NUM_MODELS = 2
CHATBOTS = [None] * NUM_MODELS


# FUNCTIONS
def respond(message: str, chat_history1: list, chat_history2: list) -> Tuple[str, list, list]:
    """Respond to the user question.

    Parameters
    ----------
    message : str
        The user question.
    chat_history1 : list
        The chat history for the first model.
    chat_history2 : list
        The chat history for the second model.
    Returns
    -------
    Tuple[str, list, list]
        Empty string to erase the previous question, 
        the updated chat history for the first model,
        the updated chat history for the second model.        
    """
    logger.info("Responding to the user question...")

    # Randomly choose two models
    chosen_models = random.sample(LLM_MODELS, 2)

    for i, model in enumerate(chosen_models):
        logger.info(f"Using the model: {model}")
        # Choose the appropriate chat history
        chat_history = chat_history1 if i == 0 else chat_history2
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
            if i == 0:
                chat_history1.append((message, response))
            else:
                chat_history2.append((message, response))
            
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
                model_name=model,
            )
            # Add metadata to the answer
            final_answer = add_metadata_to_answer(answer, metadatas=metadata, iu=True)
            if i == 0:
                chat_history1.append((message, final_answer))
            else:
                chat_history2.append((message, final_answer))

    return "", chat_history1, chat_history2


def clear_chat() -> Tuple[str, list, list]:
    """Clear the chat history.

    Parameters
    ----------
    chat_history1 : list
        The chat history for the first model.
    chat_history2 : list
        The chat history for the second model.
    Returns
    -------
    Tuple[str, list, list]
        Empty string to erase the previous question, 
        the updated chat history for the first model,
        the updated chat history for the second model.        
    """
    logger.info("Clearing the chat history...")

    fist_conv = [["Hey, j'ai besoin d'aide en Python !","Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?"]]

    return fist_conv, fist_conv

def get_vote(button_label: str):
    """Get the vote of the user.

    Parameters
    ----------
    button_label : str
        The label of the button clicked by the user.
    model_a : str
        The name of the first model.
    model_b : str
        The name of the second model.
    """
    model_a = "Model A"
    model_b = "Model B"
    if button_label == "👈  A est meilleur":
        logger.info(f"Model {model_a} vs {model_b}: {model_a}")
    elif button_label == "🤝  Les 2 se valent":
        logger.info(f"Model {model_a} vs {model_b}: Tie")
    elif button_label == "👉 B est meilleur  ":
        logger.info(f"Model {model_a} vs {model_b}: {model_b}")
    elif button_label == "👎 Les 2 sont mauvais":
        logger.info(f"Model {model_a} vs {model_b}: Both")
    else:
        logger.error("Invalid button label.")


def create_tab_battle():
    """Create the interface to discuss with the course in a battle mode."""
    with gr.Blocks(
    theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"),
    title="BioPyAssistant"
    ) as demo:
        # Define Chatbots
        with gr.Row():
            for i in range(NUM_MODELS):
                label = "Model A" if i == 0 else "Model B"
                with gr.Column():
                    CHATBOTS[i] = gr.Chatbot(
                        label=label,
                        elem_id="chatbot",
                        value=[
                            [
                                "Hey, j'ai besoin d'aide en Python !",
                                "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?",
                            ]
                        ],
                        bubble_full_width=False,
                        height=600,
                        show_copy_button=True,
                        render=True,
                        avatar_images=(("data/img/user_avatar.png"), "data/img/chatbot_avatar.png"),
                    )

        # Define the vote buttons
        with gr.Row():
            leftvote_btn = gr.Button(value="👈  A est meilleur")
            tie_btn = gr.Button(value="🤝  Les 2 se valent")
            bothbad_btn = gr.Button(value="👎  Les 2 sont mauvais")
            rightvote_btn = gr.Button(value="👉  B est meilleur")

        # Define the query textbox
        with gr.Row():            
            msg = gr.Textbox(
                placeholder="Pose moi une question sur le cours !",
                render=True,
                show_label=False,
                min_width=1200,
            )
            # Define the clear button
            clear_btn = gr.ClearButton(value="Effacer l'historique")
        
        msg.submit(respond, inputs=[msg, CHATBOTS[0], CHATBOTS[1]], outputs=[msg, CHATBOTS[0], CHATBOTS[1]])
    
        leftvote_btn.click(get_vote, inputs=[leftvote_btn])
        tie_btn.click(get_vote, inputs=[tie_btn])
        bothbad_btn.click(get_vote, inputs=[bothbad_btn])
        rightvote_btn.click(get_vote, inputs=[rightvote_btn])
        clear_btn.click(clear_chat, outputs=[CHATBOTS[0], CHATBOTS[1]])
        
    return demo



# MAIN PROGRAM
if __name__ == "__main__":
    # Load the environment variables
    load_dotenv()

    # Filter the warnings
    logger.add("file.log", level="INFO")
    
    # Create the interface to discuss with the course
    demo = create_tab_battle()

    # Launch the Gradio interface
    demo.launch(
        server_name="0.0.0.0",  # to make the app accessible from other devices
        inbrowser=True,  # to automatically opens a new tab
    )

