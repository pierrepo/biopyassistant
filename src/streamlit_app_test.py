"""Streamlit app for chatbot testing.

Usage:
======
    streamlit run src/streamlit_app_test.py

"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRAIRIES IMPORT
import openai
import streamlit as st
from loguru import logger
from langchain_core.documents import Document
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# CONSTANTS
CHROMA_PATH = "chroma_db"
PROMPT_TEMPLATE_COURSE = """
Tu es un assistant pour les tâches de question-réponse des étudiants dans un cours de programmation Python.
Tu dois fournir des réponses à leurs questions basées sur les supports de cours.
Utilise les morceaux de contexte suivants pour répondre à la question.

Question : "{question}"

Contexte : "{contexte}"


Répond à la question de manière claire et concise en français.
La réponse doit être facile à comprendre pour les étudiants.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
Si tu as besoin de plus d'informations, tu peux demander.
Si tu as besoin de clarifier la question, tu peux le demander.
"""
PROMPT_TEMPLATE_EXERCICE = """
Tu es un assistant pour les tâches de question-réponse des étudiants dans un cours de programmation Python.
Tu dois fournir des réponses à leurs questions basées sur les supports de cours.
Utilise les morceaux de contexte suivants pour répondre à la question.

Question : "{question}"

Contexte : "{contexte}"


Répond à la question de manière claire et concise en français.
La réponse doit être facile à comprendre pour les étudiants.
Si tu ne connais pas la réponse, dis simplement que tu ne sais pas.
Si tu as besoin de plus d'informations, tu peux demander.
Si tu as besoin de clarifier la question, tu peux le demander.
"""

# STREAMLIT APP
class StreamlitApp:
    """Streamlit app for chatbot testing."""
    def __init__(self):
        """Initialize the app."""
        self.title = "BioPyAssistant - OpenAi Chatbot Test 🐍"
        # Set the page configuration
        st.set_page_config(page_title="BioPyAssistant", page_icon="data/logo.webp", layout="wide")
        

    def run(self):
        """Run the app."""        
        question_type, python_level, model_name, openai_api_key = self.create_sidebar()
        vector_db = self.load_database()
        self.chat_with_bot(openai_api_key)
        

    def check_openai_api_key(self, api_key):
        client = openai.OpenAI(api_key=api_key)
        try:
            client.models.list()
            st.sidebar.success('Your OpenAI API key is valid!', icon='✅')

        except openai.AuthenticationError:
            st.sidebar.warning('Please enter a valid OpenAI API key!', icon='⚠')
            st.sidebar.info("Obtain your key from this link: [OpenAI API Keys](https://platform.openai.com/account/api-keys)")

        except openai.APIConnectionError:
            pass # Do nothing, the key is valid
        
        
    def create_sidebar(self):
        """Create the sidebar."""
        st.sidebar.title("Options")

        # Choose between question on course or exercises
        question_type = st.sidebar.radio("Question Type:", ("Course", "Exercises"))

        # Python proficiency level
        python_level = st.sidebar.selectbox("Python Proficiency Level:", ("Beginner", "Intermediate", "Advanced"))

        # Model selection
        model_name = st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo", "gpt-4"))

        # Get the OpenAI API key
        openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')
        self.check_openai_api_key(openai_api_key)

        # Insert multiple spaces
        st.sidebar.markdown("")
        st.sidebar.markdown("")
        st.sidebar.markdown("")

        # Initialize total cost if not already set
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0.0

        # Get the cost of the discussion
        counter_placeholder = st.sidebar.empty()
        counter_placeholder.write(f"Total cost of this conversation: <span style='color:#8DD3C3'>{st.session_state.total_cost:.3f}$</span>", unsafe_allow_html=True)

        return question_type, python_level, model_name, openai_api_key
        
        
    def load_database(self) -> Chroma:
        """Prepare the vector database.

        Returns
        -------
            Chroma: The prepared vector database.
        """
        logger.info("Loading the vector database.")
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-large") # define the embdding model
        # Load the database from the specified directory
        vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        logger.info(f"Chunks in the database: {vector_db._collection.count()}")
        logger.success("Vector database prepared successfully.")

        return vector_db

    
    def generate_response(self, input_text, openai_api_key):
        pass


    def chat_with_bot(self, openai_api_key):
        """Chat with the bot."""
        st.title(self.title)

        if "messages" not in st.session_state.keys(): # Initialize the chat message history
            st.session_state.messages = [
                {"role": "assistant", "content": "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?"}
            ]

        user_query = st.chat_input(placeholder="Ask me something!")


# MAIN PROGRAM
if __name__ == "__main__" :
    app = StreamlitApp()
    app.run()