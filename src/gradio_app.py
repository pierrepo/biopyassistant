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

import gradio as gr
from loguru import logger


# MODULES IMPORT
from gradio_app_chat import create_tab_chatbot
from gradio_app_quiz import create_tab_quiz


# CONSTANTS
FLAVICON_RELATIVE_PATH = "data/img/logo_round.ico"


# FUNCTIONS
def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks(
        theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"),
        title="BioPyAssistant"
    ) as demo:
        # Add a title
        gr.HTML(
            """<h1 style="font-size: 3em;"><center> 🐍 BioPyAssistant 🐍 </center></h1>"""
        )

        # Add a section for asking a question to the chatbot about the course
        with gr.Tab("Discuter avec le cours"):
            create_tab_chatbot()

        # Add a section for asking a qcm about a specific chapter
        with gr.Tab("Se tester"):
            create_tab_quiz()

    return demo


# MAIN PROGRAM
if __name__ == "__main__": 
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