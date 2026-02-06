"""Streamlit app for chatbot testing."""

import secrets
import sys
import time
from datetime import timedelta
from pathlib import Path

import loguru
import pyperclip
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from loguru import logger

from model_config_app import Settings
from query_chatbot import (
    MSGS_QUERY_NOT_RELATED,
    add_metadata_to_answer,
    contextualize_question,
    format_chat_history,
    generate_answer,
    get_metadata,
    load_database,
    search_similarity_in_database,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CSS_PATH = PROJECT_ROOT / "assets" / "style.css"
MIN_TIME_BETWEEN_REQUESTS = timedelta(seconds=2)

SUGGESTIONS = {
    ":green[:material/loop:] Comment écrire une boucle ?": "Comment écrire une boucle ?",
    ":blue[:material/list_alt:] Quelle est la différence entre une liste et un set ?": (
        "Quelle est la différence entre une liste et un set ?"
    ),
    ":orange[:material/decimal_increase:] Comment afficher un float avec 2 chiffres avec la virgule ?": (
        "Comment afficher un float avec 2 chiffres avec la virgule ?"
    ),
}


def create_logger(
    logpath: str | Path | None = None, level: str = "INFO"
) -> "loguru.Logger":
    """Create the logger with optional file logging.

    Parameters
    ----------
    logpath : str | Path | None, optional
        Path to the log file. If None, no file logging is done.
    level : str, optional
        Logging level. Default is "INFO".

    Returns
    -------
    loguru.Logger
        Configured logger instance.
    """
    # Define log format.
    logger_format = (
        "{time:YYYY-MM-DD HH:mm:ss} "
        "| <level>{level:<8}</level> "  # noqa: RUF027
        "| <level>{message}</level>"
    )
    # Remove default logger.
    logger.remove()
    # Add logger to path (if path is provided).
    if logpath:
        # Create parent directories.
        Path(logpath).parent.mkdir(parents=True, exist_ok=True)
        # Add logger to file.
        logger.opt(colors=True).add(
            logpath, format=logger_format, level="DEBUG", mode="w"
        )
    # Add logger to stdout.
    logger.add(sys.stdout, format=logger_format, level=level)
    return logger.opt(colors=True)


def apply_custom_css(css_file: Path) -> None:
    """Load and injects custom CSS into the Streamlit app."""
    if css_file.exists():
        css = css_file.read_text()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {css_file}")


@st.cache_resource
def get_vector_db(
    vector_db: Path, embedding_model_name: str, logger: "loguru.Logger" = loguru.logger
) -> Chroma:
    """Cache the vector database to prevent reloading on every rerun.

    Returns
    -------
    Chroma: The vector database containing the embedded course.
    """
    vector_db, _nb_chunks = load_database(vector_db, embedding_model_name, logger)
    return vector_db


def create_header(app_name: str) -> None:
    """Render the application header and subtitle."""
    st.markdown(
        f'<div class="app-title">{app_name}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-subtitle">
            BioPyAssistant est un assistant pédagogique pour le cours de
            <a href="https://python.sdv.u-paris.fr/" target="_blank">
                programmation Python
            </a>
            <br>
            pour les biologistes de Patrick Fuchs et Pierre Poulain.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.space(30)


def create_sidebar() -> dict[str, str]:
    """Render the Streamlit sidebar and collect user profile settings.

    Returns
    -------
    dict[str, str]
        Dictionary containing the user-selected profile information with
        the following keys:

        - ``"cursus"`` : Academic track selected by the user
          (e.g., "Licence", "Master").
        - ``"level"`` : User's python level
          (e.g., "Débutant", "Intermédiaire", "Avancé").
    """
    with st.sidebar:
        # Institutional logos
        st.logo(
            "https://u-paris.fr/wp-content/uploads/2022/03/UniversiteParisCite_logo_horizontal_couleur_RVB.png",
            size="large",
            icon_image="https://lvts.fr/wp-content/uploads/2022/03/UniversiteParis_monogramme_couleur_RVB-e1712425218876.png",
        )
        st.space("large")
        # Student profile
        st.markdown(
            '<div class="sidebar-title">🎓 Profil étudiant</div>',
            unsafe_allow_html=True,
        )
        cursus = st.pills("Cursus :material/school:", options=["Licence", "Master"])
        level = st.pills(
            "Niveau :material/sort:", options=["Débutant", "Intermédiaire", "Avancé"]
        )
        st.space(20)
        st.info(
            "Indiquez votre cursus et votre niveau pour des réponses adaptées.",
            icon="🎯",
        )
        st.space(250)
        # About section
        st.markdown(
            """
            <div class="sidebar-about">
                <strong>BioPyAssistant</strong> a été développé par
                <strong>
                    <a href="https://www.linkedin.com/in/essmay-touami/"
                    target="_blank">
                        Essmay Touami
                    </a>
                </strong>
                et
                <strong>
                    <a href="https://www.linkedin.com/in/pierrepo/"
                    target="_blank">
                        Pierre Poulain
                    </a>
                </strong>
                <br>
                dans le cadre du projet pédagogique<br>
                <em>
                    <a href="https://u-paris.fr/aap-innovation-pedagogique-2023-decouvrez-les-projets-laureats/"
                    target="_blank">
                        LLM@UPCité
                    </a>
                </em>.<br><br>
                Code source disponible sur GitHub<br>
                sous licence BSD 3-clause.
            </div>
            """,
            unsafe_allow_html=True,
        )

        #
        st.markdown(
            """
            <div class="sidebar-icons">
                <a href="https://github.com/pierrepo/biopyassistant" target="_blank">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg"
                         alt="GitHub">
                </a>
                <a href="https://python.sdv.u-paris.fr/" target="_blank">
                    <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg"
                         alt="Python">
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {"cursus": cursus, "level": level}


@st.dialog("💡 Guide d'utilisation responsable")
def show_disclaimer_dialog() -> None:
    """Display a dialog outlining responsible usage guidelines for the application."""
    st.caption("""
    ### 🧠 Gardez la main sur votre réflexion
    L'IA est un assistant, pas un expert infaillible.
    Le **copier-coller direct est déconseillé** :
    utilisez les réponses comme une base de travail que vous devez valider et enrichir
               par votre esprit critique.

    ### 🛡️ Protégez votre vie privée
    Ce service utilise des modèles externes.
    **Ne partagez jamais de données personnelles**,
    confidentielles ou sensibles dans vos échanges.

    ### 📜 Aller plus loin
    Pour adopter les bonnes pratiques, consultez la [Charte d'utilisation](#).
    """)


def create_footer() -> None:
    """Render the application footer with legal information."""
    st.html(
        """
        <div class="app-footer">
            <p>
                Cette application web n'utilise pas de cookie. Les résultats des votes sont collectés anonymement à des fins de recherche.
                <br>
                <a href="https://u-paris.fr/politique-de-confidentialite/" target="_blank" rel="noopener noreferrer">
                    Mentions légales.
                </a>
            </p>
        </div>
        """
    )


def send_telemetry(
    **kwargs,
) -> None:  # TODO: ask pierre what telemetry to log and how to log it
    """
    Record telemetry data related to user interactions with the chatbot.

    This function collects and logs usage metrics to help monitor
    application performance and user engagement. The telemetry may include,
    but is not limited to:

    - Questions submitted by users.
    - Number of requests per session.
    - Response time of the LLM.
    - Error rates or failed requests.
    - Clicks on suggested prompts or examples.
    - User feedback on responses (e.g., thumbs up / thumbs down).

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments containing additional context or metadata
        for telemetry, such as the question text, response, timestamps, or
        user identifiers.
    """
    pass


def show_feedback_controls(message_index: int, assistant_msg: str) -> None:
    """Display compact feedback controls for an assistant message.

    Parameters
    ----------
    message_index : int
        Index of the message in `st.session_state.messages` for which
        the feedback is being collected.
    assistant_msg : str
        The content of the assistant message to which the feedback controls
        are attached.
    """
    with st.container():
        option_map = {
            "is_copied": ":material/content_copy:",
            "has_voted_up": ":material/thumb_up:",
            "has_voted_down": ":material/thumb_down:",
            "has_report_msg": ":material/report:",
        }
        selection = st.pills(
            "assistant message",
            label_visibility="hidden",
            options=option_map.keys(),
            format_func=lambda option: option_map[option],
            selection_mode="single",
        )
        relevant_history = st.session_state.messages[: message_index + 1]
        if selection:
            if selection == "is_copied":
                pyperclip.copy(assistant_msg)
                st.toast("Réponse copiée !", icon="📋", duration="short")

            elif selection == "has_voted_up" or selection == "has_voted_down":
                vote_type = "👍" if selection == "has_voted_up" else "👎"
                logger.warning(f"Feedback {vote_type}")
                logger.warning(f"Message: {assistant_msg}")
                logger.debug(f"Context: {relevant_history}")
                st.toast("Merci pour votre vote !", icon="✅", duration="short")

            elif selection == "has_report_msg":
                with st.form(key=f"report-form-{message_index}", clear_on_submit=True):
                    details = st.text_area(
                        "Signaler / Ajouter un commentaire (optionnel)",
                        height=60,
                    )
                    if st.form_submit_button("Envoyer"):
                        logger.critical("Report 🚨")
                        logger.critical(f"Commentaire: {details or 'Aucun'}")
                        logger.critical(f"Message: {assistant_msg}")
                        logger.debug(f"Context: {relevant_history}")
                        st.toast(
                            "Merci pour votre signalement !", icon="⚠️", duration="short"
                        )


def display_welcome_chat() -> None:
    """Display the initial welcome chat message and suggestions."""
    st.session_state.messages = []

    # Display welcome container with chat input and suggestions
    with st.container():
        st.chat_input("Pose moi une question sur le cours...", key="initial_question")
        st.pills(
            label="Examples",
            label_visibility="collapsed",
            options=SUGGESTIONS,
            key="selected_suggestion",
        )
    # Show disclaimer for ethical use
    st.button(
        (
            "&nbsp;:small[:gray[:material/chat_error: Les réponses générées "
            " peuvent être incorrectes ou incomplètes, gardez toujours un esprit"
            " critique !]]"
        ),
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )

    # Stop execution so the chat doesn't process until a question is asked
    st.stop()


def transform_messages(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    """
    Convert Streamlit message history into a list of (User, Assistant) tuples.

    Args:
        messages: List of message dictionaries from st.session_state.

    Returns
    -------
        A list of tuples compatible with LangChain history formats.
    """
    history_tuples = []
    temp_user_msg = None

    for msg in messages:
        if msg["role"] == "user":
            temp_user_msg = msg["content"]
        elif msg["role"] == "assistant" and temp_user_msg is not None:
            history_tuples.append((temp_user_msg, msg["content"]))
            temp_user_msg = None

    return history_tuples


def generate_response(
    user_query: str, context: str, model_name: str, prompt_path: str
) -> str:
    """Generate a response to the user question.

    Parameters
    ----------
    user_query : str
        The user question.
    context : str
        The relevant context retrieved from the vector database.
    model_name : str
        The name of the LLM to use for generating the response.
    prompt_path : str
        Path to the prompt template to use for generating the response.

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
    chat_context = contextualize_question(chat_history_formatted=formatted_chat_history)

    # If no relevant document was found
    if context == []:
        logger.info("No relevant documents found in the database.")
        logger.success("Returning an automatic response without calling the model.")
        # random response betweet responses in MSGS_QUERY_NOT_RELATED
        response = secrets.choice(MSGS_QUERY_NOT_RELATED)
        return response
    else:
        # Get the metadata of the relevant documents
        metadata = get_metadata(context)
        # Generate the answer
        answer = generate_answer(
            query=user_query,
            chat_context=chat_context,
            relevant_chunks=context,
            model_name=model_name,
            prompt_path=prompt_path,
            logger_flag=False,
        )
        # Add metadata to the answer
        final_answer = add_metadata_to_answer(answer, metadatas=metadata, iu=True)

        # logger.info(f"Response generated: {final_answer}")
        # logger.success("Response generated successfully.\n")

        return final_answer


def stream_text(text: str):
    """Simulate a typing effect in the UI."""
    for token in text:
        yield token
        time.sleep(0.01)


def clear_conversation():
    st.session_state.messages = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None


def chat_with_bot(
    vector_db: Chroma, student_infos: dict, model_name: str, prompt_path: str
) -> None:
    """Handle the main chat interface with the user.

    Parameters
    ----------
    vector_db : Chroma
        Vector database used to retrieve context for the LLM responses.
    student_infos : dict[str, str]
        Dictionary containing the student's profile information:
        - "cursus" : Academic track selected by the user (e.g., "Licence", "Master").
        - "level" : User's Python or academic proficiency level
          (e.g., "Débutant", "Intermédiaire", "Avancé").
    """
    # Check if the student has provided both cursus and level
    has_student_infos = (student_infos["cursus"] and student_infos["level"]) is not None
    user_just_asked_initial_question = (
        "initial_question" in st.session_state and st.session_state.initial_question
    )
    # Determine if this is the first user interaction
    user_just_clicked_suggestion = (
        "selected_suggestion" in st.session_state
        and st.session_state.selected_suggestion
    )
    user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
    )
    # If first interaction but no student info, show a toast and block sending
    if user_first_interaction and not has_student_infos:
        st.toast(
            "Veuillez renseigner votre cursus et votre niveau pour continuer.",
            icon="⚠️",
            duration="short",
        )
        user_first_interaction = False

    # Check if there is already message histor
    has_message_history = (
        "messages" in st.session_state and len(st.session_state.messages) > 0
    )
    # --- Initial State (No conversation yet) ---
    if not user_first_interaction and not has_message_history:
        display_welcome_chat()

    # --- First step of the conversation ---
    # Show chat input at the bottom when a question has been asked.
    user_message = st.chat_input("Posez une question complémentaire...")
    if not user_message:
        if user_just_asked_initial_question:
            user_message = st.session_state.initial_question
        if user_just_clicked_suggestion:
            user_message = SUGGESTIONS[st.session_state.selected_suggestion]

        st.button(
            "Restart",
            icon=":material/refresh:",
            on_click=clear_conversation,
        )

    # --- Active Chat History ---
    # Display chat messages from history as speech bubbles.
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.container()  # Fix ghost message bug.

            st.markdown(message["content"])
            if message["role"] == "assistant":
                show_feedback_controls(i, message["content"])

    # When the user posts a message...
    if user_message:
        # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
        # display math). The line below fixes it.
        user_message = user_message.replace("$", r"\$")

        # Display message as a speech bubble.
        with st.chat_message("user"):
            st.text(user_message)

        # Display assistant response as a speech bubble.
        with st.chat_message("assistant"):
            # Search for relevant context in the vector database.
            with st.spinner("En recherche de contexte..."):
                context = search_similarity_in_database(
                    vector_db=vector_db,
                    user_query=user_message,
                    # user_level=student_infos["level"],
                    # user_cursus=student_infos["cursus"],
                    logger_flag=False,
                )

            with st.spinner("En réflexion..."):
                response = generate_response(
                    user_message, context, model_name, prompt_path
                )
                rep_stream = stream_text(response)

            # Put everything after the spinners in a container to fix the
            # ghost message bug.
            with st.container():
                # Stream the LLM response.
                response = st.write_stream(rep_stream)

                # Add messages to chat history.
                st.session_state.messages.append(
                    {"role": "user", "content": user_message}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

                # Show feedback controls (copy, thumbs up/down, report)
                show_feedback_controls(len(st.session_state.messages) - 1, response)
                # Send telemetry data for logging and analysis
                # (to be implemented in the future)
                send_telemetry(question=user_message, response=response)


def main():
    """Run the BioPyAssistant Streamlit app."""
    # Load environment variables
    load_dotenv()
    # Initialize settings from `config_app.toml`
    settings = Settings()
    # Configure logger once
    if "welcome_logged" not in st.session_state and not settings.log_path.exists():
        logger = create_logger(settings.log_path)
        logger.info(
            f"<fg #e64c7a>👋 Welcome to BioPyAssistant v{settings.app_version}!</>"
        )
        if settings.app_description:
            logger.info(f"<fg #303030><i>{settings.app_description}\n</i></>")
        st.session_state["welcome_logged"] = True

    # Set the page configuration
    st.set_page_config(
        page_title=settings.app_name, page_icon="data/img/logo_round.webp"
    )
    # Apply the css style
    apply_custom_css(settings.css_path)

    # Create the header
    create_header(settings.app_name)

    # Create a sidebar and get the user inputs
    student_infos = create_sidebar()

    # Create footer
    create_footer()

    # Load the vector database one
    vector_db = get_vector_db(
        settings.vector_database_path, settings.llm.embedding_model_name
    )

    # Chat with the bot
    chat_with_bot(
        vector_db, student_infos, settings.llm.llm_model_name, settings.llm.prompt_path
    )


if __name__ == "__main__":
    main()
