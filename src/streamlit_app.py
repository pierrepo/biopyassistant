"""Streamlit app for chatbot testing.

Usage:
======
    streamlit run src/streamlit_app.py

"""

# METADATA
__authors__ = ("Pierre Poulain", "Essmay Touami")
__contact__ = "pierre.poulain@u-paris.fr"
__copyright__ = "BSD-3 clause"
__date__ = "2024"
__version__ = "1.0.0"


# LIBRAIRIES IMPORT
import streamlit as st

# MODULES IMPORT
from query_chatbot import load_database, search_similarity_in_database, get_metadata, generate_prompt, predict_response, adding_metadatas_to_response


def create_header() -> None:
    st.markdown(
        f"""
        <div style="padding:20px;border-radius:10px;">
            <h1 style="color:white;text-align:center;font-family:"Monoton", sans-serif;font-size:5em;">🐍 BioPyAssistant 🐍</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")


def insert_multiple_spaces(n: int = 1) -> None:
    """Insert multiple spaces in the streamlit app."""
    for _ in range(n):
        st.sidebar.markdown("")


def create_sidebar():
        """Create the sidebar."""
        st.sidebar.title("Options")

        # Choose between question on course or exercises
        question_type = st.sidebar.radio("Type de Question :", ("Cours", "Exercices"))

        # Python proficiency level
        python_level = st.sidebar.selectbox("Niveau de Maîtrise de Python :", ("Débutant", "Intermédiaire", "Avancé"))

        # Model selection
        model_name = st.sidebar.radio("Choisir un modèle :", ("gpt-3.5-turbo", "gpt-4"))

        # Insert multiple spaces
        insert_multiple_spaces(3)

        # Initialize total cost if not already set
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0.0

        # Get the cost of the discussion
        counter_placeholder = st.sidebar.empty()
        counter_placeholder.write(f"Coût total de cette conversation : <span style='color:#8DD3C3'>" +
                                  f"{st.session_state.total_cost:.3f}$</span>", unsafe_allow_html=True)
        # Insert multiple spaces
        insert_multiple_spaces(3)

        # Clear the conversation
        if st.sidebar.button("Effacer la conversation"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?"}
            ]
            st.session_state.total_cost = 0.0

        return question_type, python_level, model_name


def generate_response(user_query, vector_db, question_type, python_level, model_name):
    """Generate a response to the user question."""

    # Search for relevant documents in the database
    results = search_similarity_in_database(vector_db, user_query)
    
    # Get the metadata of the most similar document
    metadatas = get_metadata(results)

    # Generate a prompt for the AI model
    prompt = generate_prompt(results, user_query, python_level, question_type)

    # Predict the response using the AI model
    response_from_model = predict_response(prompt, model_name)
    
    # Add metadata to the response
    response = adding_metadatas_to_response(response_from_model, metadatas)

    return response


def chat_with_bot(vector_db, question_type, python_level, model_name) -> None:
    """Chat with the bot"""
    # Initialize the chat with a welcome message
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?"}
        ]

    # Get the user question
    if question_type == "Cours":
        if user_query := st.chat_input("Pose moi une question sur le cours !"): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
    else:
        if user_query := st.chat_input("Indique moi l'exercice sur lequel tu veux de l'aide !"): # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(user_query, vector_db, question_type, python_level, model_name)
                st.write(response)
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message) # Add response to message history
        

     

def main():
    # Set the page configuration
    st.set_page_config(page_title="BioPyAssistant", page_icon="data/logo.webp")

    # Create the header
    create_header()
    
    # Create a sidebar and get the user inputs
    question_type, python_level, model_name = create_sidebar()

    # Load the vector database
    vector_db = load_database()

    # Chat with the bot
    chat_with_bot(vector_db, question_type, python_level, model_name)


if __name__ == "__main__":
    main()
