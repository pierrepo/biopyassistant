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
import random

import streamlit as st
from loguru import logger


# MODULES IMPORT
from query_chatbot import load_database, search_similarity_in_database, get_metadata, format_chat_history, contextualize_question,  generate_answer, add_metadata_to_answer, MSGS_QUERY_NOT_RELATED, OPENAI_MODEL_NAME, CHROMA_PATH


# FUNCTIONS
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
        st.sidebar.title("Options avancées")

        # Python proficiency level
        python_level = st.sidebar.selectbox("Niveau de Maîtrise de Python :", ("Débutant", "Intermédiaire", "Avancé"))

        # Insert multiple spaces
        insert_multiple_spaces(3)

        # Initialize total cost if not already set
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0.0

        # Insert multiple spaces
        insert_multiple_spaces(3)

        # Clear the conversation
        if st.sidebar.button("Effacer la conversation"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?"}
            ]
            st.session_state.total_cost = 0.0

        return python_level


def transform_messages(messages: list[dict]) -> list[tuple]:
    """Transform the messages into a list of tuples.

    Parameters
    ----------
    messages : list[dict]
        The chat history messages.
    
    Returns
    -------
    list[tuple]
        The chat history messages as a list of tuples.
    """
    logger.info("Transforming the messages into a list of tuples.")
    messages_tuples = []
    user_message = None

    # Ignorer le premier message automatique de l'assistant
    start_index = 1 if messages[0]['role'] == 'assistant' else 0
    
    for message in messages[start_index:]:
        if message['role'] == 'user':
            user_message = message['content']
        elif message['role'] == 'assistant' and user_message is not None:
            messages_tuples.append((user_message, message['content']))
            user_message = None  # Réinitialiser le message de l'utilisateur après avoir créé le tuple
    
    return messages_tuples


def generate_response(user_query: str, vector_db, python_level: str) -> str:
    """Generate a response to the user question.
    
    Parameters
    ----------
    user_query : str
        The user question.
    python_level : str
        The Python level of the user.

    Returns
    -------
    response : str
        The response predicted by the model.
    """
    # Transform the chat history into a list of tuples
    chat_history = transform_messages(st.session_state.messages)
    # Format the chat history for the model
    formatted_chat_history = format_chat_history(chat_history, len_history=10)

    # Contextualize the user question with the chat history
    query_contextualized = contextualize_question(user_query=user_query, chat_history_formatted=formatted_chat_history, model_name=OPENAI_MODEL_NAME)
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


def chat_with_bot(vector_db, python_level) -> None:
    """Chat with the bot"""
    # Initialize the chat with a welcome message
    if "messages" not in st.session_state.keys(): # Initialize the chat message history
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour, je suis BioPyAssistant, ton assistant pour répondre à tes questions sur Python. Comment puis-je t'aider ?"}
        ]

    # Get the user question
    if user_query := st.chat_input("Pose moi une question sur le cours !"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("En réflexion..."):
                response = generate_response(user_query, vector_db, python_level)
                st.write(response)

                # Add the response to the chat history
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)


def main():
    # Set the page configuration
    st.set_page_config(page_title="BioPyAssistant", page_icon="data/logo_round.webp")

    # Create the header
    create_header()
    
    # Create a sidebar and get the user inputs
    python_level = create_sidebar()

    # Load the vector database
    vector_db = load_database(CHROMA_PATH)[0]

    # Chat with the bot
    chat_with_bot(vector_db, python_level)


# MAIN PROGRAM
if __name__ == "__main__":
    main()
