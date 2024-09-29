"""Gradio application of BioPyAssistant.

It allows users to interact with the course in different ways, such as:
- Asking questions to the chatbot.
- Asking questions to the chatbot in a battle mode.
- Answering quizzes about the course.

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

from dotenv import load_dotenv
import gradio as gr
from loguru import logger

# Load the environment variables with LLM api keys.
logger.info(f"Loading API keys...")
load_dotenv()

# MODULES IMPORT
from gradio_app_chat import create_tab_chatbot
from gradio_app_quiz import create_tab_quiz
from gradio_app_battle import create_tab_battle


# CONSTANTS
FLAVICON_RELATIVE_PATH = "data/img/logo_round.ico"
CSS = """
#correct-answer
    {background-color: #46772E}
#wrong-answer
    {background-color: #F28C00}
#footer {
    position: absolute;
    bottom: 0;
    transform: scale(0.9);
}

/* Footer becomes relative for small screens */
@media only screen and (max-width: 800px) {
  #footer {
    /* display: none; */
    position: relative;
  } 
}

@media only screen and (max-height: 1000px) {
  #footer {
    /* display: none; */
    position: relative;
  } 
}
"""

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Default(primary_hue="emerald", secondary_hue="emerald"),
    title="BioPyAssistant",
    css=CSS, fill_height=True
) as demo:
    # Add a title
    gr.HTML(
        """<h2><center>🐍 BioPyAssistant 🐍</center></h2>"""
    )

    # Add a section for asking a question to the chatbot about the course
    with gr.Tab("Discuter avec le cours"):
        create_tab_chatbot()

    # Add a section for asking a question to the chatbot about the course but in a battle mode
    with gr.Tab("Discuter avec le cours (battle)"):
        create_tab_battle()

    # Add a section for asking a qcm about a specific chapter
    # with gr.Tab("Se tester"):
    #     create_tab_quiz()
    
    # Add footer.
    with gr.Row(elem_id="footer", equal_height=False):
        with gr.Column(scale=1):
            gr.HTML("<img src='https://u-paris.fr/wp-content/uploads/2022/03/Universite_Paris-Cite-logo.jpeg' width='200px'>")
        with gr.Column(scale=3):
            gr.Markdown("""
            [Mentions légales](https://u-paris.fr/politique-de-confidentialite/).
            Cette application web n'utilise pas de cookie.
            Les résultats des votes sont collectés anonymement à des fins de recherche.
            
            BioPyAssistant a été développé par [Essmay Touami](https://www.linkedin.com/in/essmay-touami/) et [Pierre Poulain](https://www.linkedin.com/in/pierrepo/) dans le cadre du projet pédagogique [LLM@UPCité](https://u-paris.fr/aap-innovation-pedagogique-2023-decouvrez-les-projets-laureats/).
            Le code source est disponible sur [GitHub](https://github.com/pierrepo/biopyassistant) sous licence BSD 3-clause.
            """)


# MAIN PROGRAM
if __name__ == "__main__":
    # Get the favicon path
    FLAVICON_PATH = os.path.abspath(FLAVICON_RELATIVE_PATH)
    logger.info(f"Flavicon path: {FLAVICON_PATH}")

    # Launch the Gradio interface
    demo.launch(
        favicon_path=FLAVICON_PATH,  # to add a favicon
        server_name="0.0.0.0",  # to make the app accessible from other devices
        server_port=8080
    )