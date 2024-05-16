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
from query_chatbot import load_database, search_similarity_in_database, get_metadata, generate_prompt, predict_response, adding_metadatas_to_response, MSGS_QUERY_NOT_RELATED


# CONSTANTS
VECTOR_DB_PATH = "chroma_db"
OPENAI_MODEL_NAME = "gpt-3.5-turbo"


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


def generate_response(user_query, vector_db, python_level):
    """Generate a response to the user question."""

    # Search for relevant documents in the database
    relevant_chunks = search_similarity_in_database(vector_db, user_query)

    # If relevant document was found, return the response from the model
    if relevant_chunks != []:
        # Generate a prompt for the AI model
        prompt, nb_tokens_in_prompt = generate_prompt(relevant_chunks, user_query, python_level)

        # Add the history of the conversation to the prompt
        if "messages" in st.session_state.keys():
            prompt += "Conversation History:"
            for message in st.session_state.messages:
                if message["role"] == "user":
                    prompt += f"\nUser: {message['content']}"
                elif message["role"] == "assistant":
                    prompt += f"\nAssistant: {message['content']}"

        # Predict the response using the AI model and get the number of tokens in the response
        response_from_model, nb_tokens_in_response = predict_response(prompt, OPENAI_MODEL_NAME)

        # Get the metadata of the most similar document
        metadatas = get_metadata(relevant_chunks)
        
        # Add metadata to the response
        response_with_metadata = adding_metadatas_to_response(response_from_model, metadatas, iu=True)

        return response_with_metadata, nb_tokens_in_prompt, nb_tokens_in_response

    else: # If no relevant document was found
        return MSGS_QUERY_NOT_RELATED[0], 0, 0


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
                response, nb_tokens_in_prompt, nb_tokens_in_response = generate_response(user_query, vector_db, python_level)
                st.write(response)

                # Add the response to the chat history
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

                # Update the total cost of the conversation
                if OPENAI_MODEL_NAME == "gpt-3.5-turbo":
                    st.session_state.total_cost += 0.0000005 * nb_tokens_in_prompt
                    st.session_state.total_cost += 0.0000015 * nb_tokens_in_response
                elif OPENAI_MODEL_NAME == "gpt-4":
                    st.session_state.total_cost += 0.00003 * nb_tokens_in_prompt
                    st.session_state.total_cost += 0.00006 * nb_tokens_in_response
                print(f"Total cost: {st.session_state.total_cost:.5f}")


def main():
    # Set the page configuration
    st.set_page_config(page_title="BioPyAssistant", page_icon="data/logo_round.webp")

    # Create the header
    create_header()
    
    # Create a sidebar and get the user inputs
    python_level = create_sidebar()

    # Load the vector database
    vector_db = load_database(VECTOR_DB_PATH)[0]

    # Chat with the bot
    chat_with_bot(vector_db, python_level)


# MAIN PROGRAM
if __name__ == "__main__":
    main()
