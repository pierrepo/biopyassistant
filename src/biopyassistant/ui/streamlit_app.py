"""Streamlit app for chatbot."""

import secrets
import time
from pathlib import Path

import loguru
import pyperclip
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from loguru import logger

from biopyassistant.core.messages import SUGGESTIONS
from biopyassistant.core.query_chatbot import (
    MSGS_QUERY_NOT_RELATED,
    generate_answer,
    load_database,
    search_similarity_in_database,
)
from biopyassistant.logger import create_logger
from biopyassistant.models.app_settings import Settings
from biopyassistant.models.course import CourseLevel


def apply_custom_css(css_file: Path) -> None:
    """Load and injects custom CSS into the Streamlit app."""
    if css_file.exists():
        css = css_file.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        logger.warning(f"CSS file not found at {css_file}")


@st.cache_resource
def get_vector_db(
    vector_db_path: Path,
    embeddings_model_name: str,
    provider_embeddings_name: str,
    _logger: "loguru.Logger" = loguru.logger,
) -> Chroma:
    """Cache the vector database to prevent reloading on every rerun.

    Returns
    -------
    Chroma: The vector database containing the embedded course.
    """
    return load_database(
        vector_db_path, embeddings_model_name, provider_embeddings_name, _logger
    )


def create_header(app_name: str) -> None:
    """Render the application header and subtitle."""
    st.markdown(
        f'<div class="app-title">{app_name}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-subtitle">
            Un assistant pédagogique pour le cours de
            <a href="https://python.sdv.u-paris.fr/" target="_blank">
                programmation Python pour les biologistes
            </a>
            de Patrick Fuchs et Pierre Poulain.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.space(30)


def create_footer() -> None:
    """Render the application footer with legal information."""
    with st.container(key="app-footer"):
        st.markdown(
            """
            Les interactions avec cet assistant sont collectées anonymement
            à des fins de recherche.<br />
            Ce site web n'utilise pas de cookie.
            <a href="https://u-paris.fr/politique-de-confidentialite/" target="_blank">
            Mentions légales
            </a>.
            """,
            unsafe_allow_html=True,
        )


def on_level_change(logger: "loguru.Logger" = loguru.logger) -> None:
    """Handle level selection change: log and reset conversation."""
    clear_conversation(logger)
    logger.info(f"User selected level: {st.session_state.selected_level}")


def create_sidebar(
    course_levels: dict[str, CourseLevel],
    logger: "loguru.Logger" = loguru.logger,
) -> str:
    """Render the Streamlit sidebar and collect the selected level.

    Parameters
    ----------
    course_levels : dict[str, CourseLevel]
        Mapping from internal level name to CourseLevel objects.
    logger : loguru.Logger
        Logger instance for logging user interactions and application events.

    Returns
    -------
    str
       Internal level identifier selected by the user.
    """
    with st.sidebar:
        # Institutional logos
        st.logo(
            # Transparent image to preserve spacing
            "assets/1x1.png",
            size="large",
            # Display the University of Paris logo when sidebar is collapsed
            icon_image="assets/UniversiteParis_monogramme_couleur_RVB.png",
        )
        # Display the University of Paris logo in the sidebar when expanded
        st.image("assets/UniversiteParisCite_logo_horizontal_couleur_RVB.png")
        # Student profile
        # Level selection pills from course_levels
        selected_level = st.radio(
            label="**🎓 Sélectionnez votre cours :**",
            # We use the internal level name as the option value
            options=list(course_levels.keys()),
            # But display the user-friendly name from the CourseLevel object
            format_func=lambda key: course_levels[key].display_name,
            key="selected_level",
            # Select the first level by default
            index=0,
            on_change=on_level_change,
            args=(logger,),
        )
        # About section at the bottom of the sidebar.
        footer = st.sidebar.container(key="sidebar-footer")
        footer.markdown(
            """
            **BioPyAssistant** est développé par
            [Essmay Touami](https://www.linkedin.com/in/essmay-touami/)
            et
            [Pierre Poulain](https://www.linkedin.com/in/pierrepo/)
            dans le cadre du projet pédagogique
            [LLM@UPCité](https://u-paris.fr/aap-innovation-pedagogique-2023-decouvrez-les-projets-laureats/).

            Le code source est disponible sur [GitHub](https://github.com/pierrepo/biopyassistant)
            sous licence BSD 3-clause.
            """
        )
    return selected_level


@st.dialog(
    "💡 Guide d'utilisation responsable d'un assistant conversationnel pédagogique"
)
def show_disclaimer_dialog() -> None:
    """Display a dialog outlining responsible AI usage guidelines."""
    st.caption("""
    ### 🧠 Conservez votre esprit critique
    Cet assistant n'est pas infallible.
    Il peut parfois générer des réponses incorrectes.
    Soyez toujours vigilants et critiques quant aux réponses fournies.

    ### 🔗 Vérifiez les sources
    Vérifiez dans le cours que les réponses suggérées sont correctes.
    À la fin de chaque réponse, des liens vous emmènent directement
    vers les rubriques du cours pertinentes.

    ### 🛡️ Ne partagez pas d'informations sensibles
    Cet assistant utilise des modèles externes.
    Ne partagez jamais de données personnelles,
    confidentielles ou sensibles dans vos échanges avec cet assistant.
    """)


def clear_conversation(logger: "loguru.Logger" = loguru.logger) -> None:
    """Clear the conversation history and reset session state."""
    logger.info("User restarted the conversation.")
    st.session_state.messages = []
    st.session_state.initial_question = None
    st.session_state.selected_suggestion = None


def display_welcome_chat() -> None:
    """Display the initial welcome chat message and suggestions."""
    st.session_state.messages = []
    st.session_state.token_usage = {"input_tokens": 0, "output_tokens": 0}

    # Display welcome container with chat input and suggestions
    with st.container():
        st.chat_input("Posez-moi une question sur le cours...", key="initial_question")
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
            " peuvent être incorrectes ou incomplètes, conservez votre esprit"
            " critique !]]"
        ),
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )
    # Stop execution so the chat doesn't process until a question is asked
    st.stop()


def transform_messages(messages: list[dict[str, str]]) -> str:
    """
    Convert Streamlit message history into a list of (User, Assistant) tuples.

    Args:
        messages: List of message dictionaries from st.session_state.

    Returns
    -------
    str
        A formatted string representing the chat history, where each message
        is prefixed by its role (user or assistant) for clarity in the prompt.
    """
    formatted_history = ""
    for msg in messages:
        formatted_history += f"{msg['role']}: {msg['content']}\n"

    return formatted_history


def extract_sources(
    relevant_chunks: list[Document],
) -> list[dict[str, str]]:
    """Extract unique source labels and URLs from retrieved document chunks.

    Parameters
    ----------
    relevant_chunks : list[Document]
        Retrieved document chunks containing metadata.

    Returns
    -------
    list[dict[str, str]]
        Unique sources formatted as dictionaries with "label" and "url".
    """
    # Map each unique label to its corresponding URL
    unique_sources = {}

    for document in relevant_chunks:
        # Extract metadata fields
        metadata = document.metadata
        file_name = metadata["file_name"]
        chapter_name = metadata["chapter_name"]
        section_name = metadata.get("section_name", "")
        subsection_name = metadata.get("subsection_name", "")
        subsubsection_name = metadata.get("subsubsection_name", "")
        section_url = metadata.get("url", "")
        # Select the most specific available section level
        detailed_section = subsubsection_name or subsection_name or section_name
        # Skip entries without a valid URL
        if not section_url:
            continue
        # Build the display label depending on file type
        if file_name.startswith("annexe"):
            source_label = f"Annexe **{chapter_name}**"
        else:
            source_label = f"Chapitre **{chapter_name}**"
        # Append section detail if available
        if detailed_section:
            source_label += f", rubrique **{detailed_section}**"
        # Store unique label → URL mapping
        unique_sources[source_label] = section_url

    # Convert mapping into a list of dictionaries
    return [{"label": label, "url": url} for label, url in unique_sources.items()]


def generate_response(
    user_query: str,
    context: str,
    model_name: str,
    provider_llm_name: str,
    prompt_path: str,
    student_level: str | None,
    level_relevant_chapters: list[str],
    course_level_infos: dict[str, CourseLevel],
) -> tuple[str, int, int]:
    """Generate a response to the user question.

    Parameters
    ----------
    user_query : str
        The user question.
    context : str
        The relevant context retrieved from the vector database.
    model_name : str
        The name of the LLM to use for generating the response.
    provider_llm_name : str
        The name of the LLM provider (e.g., "openrouter") to use for
        generating the response.
    prompt_path : str
        Path to the prompt template to use for generating the response.
    student_level : str | None
        The user's Python or academic proficiency level
        (e.g., "beginner", "intermediate", "advanced").
    level_relevant_chapters : list[str]
        List of course levels available for filtering relevant context
        based on the user's selected level.
    course_level_infos : dict[str, CourseLevel]
        Mapping from internal level name to CourseLevel objects, used to get
        the prompt path for the selected student level.

    Returns
    -------
    tuple[str, int, int]
        The generated answer, the number of input tokens, and the number
        of output tokens.
    """
    # Transform the chat history for the prompt
    chat_history = transform_messages(st.session_state.messages)

    # If no relevant document was found
    if context == []:
        logger.info("No relevant documents found in the database.")
        logger.success("Returning an automatic response without calling the model.")
        # random response betweet responses in MSGS_QUERY_NOT_RELATED
        response = secrets.choice(MSGS_QUERY_NOT_RELATED)
        return response, 0, 0
    else:
        # Generate the answer
        answer, input_tokens, output_tokens = generate_answer(
            query=user_query,
            provider_llm_name=provider_llm_name,
            chat_history=chat_history,
            relevant_chunks=context,
            model_name=model_name,
            prompt_file=prompt_path,
            user_level=student_level,
            level_relevant_chapter_ids=level_relevant_chapters,
            course_level_infos=course_level_infos,
            logger=logger,
        )

        return answer, input_tokens, output_tokens


def stream_text(text: str):
    """Simulate a typing effect in the UI.

    Parameters
    ----------
    text : str
        The text to be streamed.

    Yields
    ------
    str
        The full text after streaming.
    """
    for token in text:
        yield token
        time.sleep(0.01)


def format_recent_user_queries(
    messages: list[dict[str, str]],
    nb_questions: int,
    logger: "loguru.Logger" = loguru.logger,
) -> str:
    """
    Concatenate all user questions into a single string.

    Parameters
    ----------
    messages : list[dict[str, str]]
        The chat history, where each message has a "role" and "content".
    nb_questions : int
        The number of most recent user questions to include in the concatenated string.
    logger : loguru.Logger
        Logger instance for logging the formatted user queries.

    Returns
    -------
    str
        A single string containing all user questions, separated by newlines.
    """
    user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
    # Keep only the last nb_questions questions
    # This allows the model to retrieve relevant chunks
    # based on the recent conversation, not just the last question.
    recent_user_messages = user_messages[-nb_questions:]
    queries = "\n".join(recent_user_messages)
    logger.debug("Most recent user queries for vector search:")
    for query in recent_user_messages:
        logger.debug(f"- {query}")
    return queries


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
        # Define the feedback options with corresponding icons
        option_map = {
            "is_copied": ":material/content_copy:",
            "has_voted_up": ":material/thumb_up:",
            "has_voted_down": ":material/thumb_down:",
            "has_report_msg": ":material/report:",
        }
        selection = st.pills(
            "assistant message options",
            label_visibility="hidden",
            options=option_map.keys(),
            format_func=lambda option: option_map[option],
            key=f"feedback_pills_{message_index}",
        )
        relevant_history = st.session_state.messages[: message_index + 1]
        # Handle each feedback option accordingly
        if selection:
            # Copy the assistant message to clipboard
            if selection == "is_copied":
                pyperclip.copy(assistant_msg)
                st.toast("Réponse copiée !", icon="📋", duration="short")

            # Log thumbs up/down feedback with message content and context for analysis
            elif selection == "has_voted_up" or selection == "has_voted_down":
                logger.warning(f"Feedback {selection}")
                logger.warning(f"Message: {assistant_msg}")
                logger.debug(f"Context: {relevant_history}")
                st.toast("Merci pour votre vote !", icon="✅", duration="short")

            # Handle report message option with a form to collect additional details
            elif selection == "has_report_msg":
                with st.form(key=f"report-form-{message_index}", clear_on_submit=True):
                    details = st.text_area(
                        "Signaler / Ajouter un commentaire (optionnel)",
                        height=60,
                    )
                    if st.form_submit_button("Envoyer"):
                        logger.critical(
                            "Report: User reported an issue with the "
                            "assistant's response."
                        )
                        logger.critical(f"Comments: {details or 'Aucun'}")
                        logger.critical(f"Message: {assistant_msg}")
                        logger.debug(f"Context: {relevant_history}")
                        st.toast(
                            "Merci pour votre signalement !", icon="⚠️", duration="short"
                        )


def avg_chars_in_responses(messages: list[dict[str, str]], role: str) -> int:
    """
    Calculate the average number of characters in all responses of a specific role.

    Parameters
    ----------
    messages : list[dict[str, str]]
        The chat history, where each message has a "role" and "content".
    role : str
        The role of the messages to consider (e.g., "assistant").

    Returns
    -------
    int
        Average number of characters in messages of the specified role.
    """
    role_messages = [msg for msg in messages if msg["role"] == role]
    if not role_messages:
        return 0
    total_chars = sum(len(msg["content"]) for msg in role_messages)
    return total_chars // len(role_messages)


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
    messages = st.session_state.get("messages", [])
    input_tokens = st.session_state.get("token_usage", {}).get("input_tokens", 0)
    output_tokens = st.session_state.get("token_usage", {}).get("output_tokens", 0)
    logger.debug("-----------------")
    logger.debug("Session summary:")

    # User questions
    user_questions = [msg["content"] for msg in messages if msg["role"] == "user"]
    logger.debug(f"Total questions asked: {len(user_questions)}")
    logger.debug(
        "Average characters in user questions: "
        f"{avg_chars_in_responses(messages, 'user')}"
    )
    # Assistant responses
    logger.debug(
        "Average characters in assistant responses: "
        f"{avg_chars_in_responses(messages, 'assistant')}"
    )
    # Total tokens
    logger.debug(f"Total input tokens: {input_tokens}")
    logger.debug(f"Total output tokens: {output_tokens}")
    logger.debug(f"Total tokens: {input_tokens + output_tokens}")
    logger.debug("-----------------")


def _render_sources_buttons(sources: list[dict[str, str]]) -> None:
    """Display clickable source buttons."""
    st.markdown("Pour plus d'informations, consultez les rubriques du cours :")

    for source in sources:
        st.link_button(
            label=source["label"],
            url=source["url"],
            type="secondary",
            icon=":material/open_in_new:",
        )


def _render_previous_messages() -> None:
    """Render existing chat history."""
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                # Add buttons for sources if they exist in the message
                if message.get("sources"):
                    _render_sources_buttons(message["sources"])
                # Show feedback controls (copy, thumbs up/down, report)
                show_feedback_controls(index, message["content"])


def chat_with_bot(
    vector_db: Chroma,
    embeddings_model_name: str,
    provider_embeddings_name: str,
    student_level: str | None,
    level_relevant_chapters: list[str],
    course_level_infos: dict[str, CourseLevel],
    model_name: str,
    provider_llm_name: str,
    prompt_path: Path,
    logger: "loguru.Logger" = loguru.logger,
) -> None:
    """Handle the main chat interface with the user.

    Parameters
    ----------
    vector_db : Chroma
        Vector database used to retrieve context for the LLM responses.
    embeddings_model_name : str
        The name of the embeddings model used for the vector database.
    provider_embeddings_name : str
        The name of the embeddings provider (e.g., "openai") used for the vector
        database.
    level_relevant_chapters : list[str]
        List of course levels available for filtering relevant context
        based on the user's selected level.
    course_level_infos : dict[str, CourseLevel]
        Mapping from internal level name to CourseLevel objects, used to get
        the prompt path for the selected student level.
    student_level : str | None
        The user's Python or academic proficiency level
        (e.g., "beginner", "intermediate", "advanced").
    model_name : str
        The name of the LLM to use for generating responses.
    provider_llm_name : str
        The name of the LLM provider (e.g., "openrouter") to use for
        generating responses.
    prompt_path : Path
        Path to the prompt template to use for generating responses.
    logger : loguru.Logger
        Logger instance for logging user interactions and application events.
    """
    # Check if the student has provided level information
    has_student_infos = student_level is not None
    # Determine if this is the first user interaction
    user_just_asked_initial_question = (
        "initial_question" in st.session_state and st.session_state.initial_question
    )
    user_just_clicked_suggestion = (
        "selected_suggestion" in st.session_state
        and st.session_state.selected_suggestion
    )
    user_first_interaction = (
        user_just_asked_initial_question or user_just_clicked_suggestion
    )
    # If first interaction but no student info, show a toast and block sending
    if user_first_interaction and not has_student_infos:
        logger.warning("User tried to ask a question without providing level info.")
        st.toast(
            "Veuillez indiquez votre niveau pour des réponses adaptées.",
            icon="⚠️",
            duration="short",
        )
        user_first_interaction = False

    # Check if there is already message history
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
            logger.info("User just asked the initial question")
        if user_just_clicked_suggestion:
            user_message = SUGGESTIONS[st.session_state.selected_suggestion]
            logger.info("User has clicked a suggestion question")
    # Show a button to restart the conversation and clear the history
    st.button(
        "Recommencer la conversation",
        icon=":material/refresh:",
        on_click=lambda: clear_conversation(logger),
        key="restart_button",
    )

    # Display chat messages from history as speech bubbles.
    _render_previous_messages()

    # When the user posts a message...
    if user_message:
        logger.info(f"User asked: {user_message}")
        # Streamlit's Markdown engine interprets "$" as LaTeX code (used to
        # display math). The line below fixes it.
        user_message = user_message.replace("$", r"\$")

        # Display message as a speech bubble.
        with st.chat_message("user"):
            st.text(user_message)
            # Add messages to chat history.
            st.session_state.messages.append({"role": "user", "content": user_message})

        # Display assistant response as a speech bubble.
        with st.chat_message("assistant"):
            # Concatenate all user questions from the chat history
            # to allow the model to retrieve relevant chunks based on
            # the recent conversation, not just the last question.
            user_queries = format_recent_user_queries(
                st.session_state.messages, nb_questions=3, logger=logger
            )
            # Search for relevant context in the vector database.
            context = search_similarity_in_database(
                vector_db=vector_db,
                user_query=user_queries,
                provider_embeddings_name=provider_embeddings_name,
                embedding_model=embeddings_model_name,
                logger=logger,
            )

            with st.spinner("En réflexion..."):
                # Generate the LLM response based on the user question
                # and retrieved context.
                response, input_tokens, output_tokens = generate_response(
                    user_message,
                    context,
                    model_name,
                    provider_llm_name,
                    prompt_path,
                    student_level,
                    level_relevant_chapters,
                    course_level_infos,
                )
                # Simulate streaming the response token by token for a typing effect.
                rep_stream = stream_text(response)

            # Stream the LLM response.
            response = st.write_stream(rep_stream)
            # Extract the sources from the retrieved context
            sources = extract_sources(context)
            # To display them as buttons
            if response and sources:
                _render_sources_buttons(sources)
            # Add the assistant response to the chat history.
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources}
            )
            # Show feedback controls for the new assistant message
            show_feedback_controls(len(st.session_state.messages) - 1, response)
            # Add token usage for this interaction to the session state total
            st.session_state.token_usage["input_tokens"] += input_tokens
            st.session_state.token_usage["output_tokens"] += output_tokens
            send_telemetry()


def main():
    """Run the BioPyAssistant Streamlit app."""
    # Load environment variables
    load_dotenv()
    # Initialize settings from `config_app.toml`
    settings = Settings()
    # Configure logger
    logger = create_logger(settings.log_path, ui_logger=True)
    # Display welcome message in logs only on the first run of the app
    if "welcome_logged" not in st.session_state:
        logger.info(f"👋 Welcome to BioPyAssistant v{settings.app_version}!")
        logger.info(f"{settings.app_description}\n")
        st.session_state["welcome_logged"] = True
    # Set the page configuration
    st.set_page_config(
        page_title=settings.app_name, page_icon="data/img/logo_round.webp"
    )
    # Apply the css style
    apply_custom_css(settings.css_path)
    # Create the header
    create_header(settings.app_name)
    # Create footer
    create_footer()
    # Create a sidebar and get the student level infos
    course_level_infos = settings.course_levels
    student_level = create_sidebar(settings.course_levels, logger=logger)
    # Get the chapters relevant to the selected level
    # to filter the vector database search
    level_relevant_chapters = {}
    prompt_path = None
    if student_level:
        level_relevant_chapters = course_level_infos[student_level].chapters
        prompt_path = course_level_infos[student_level].prompt_file
    # Load the vector database once
    vector_db = get_vector_db(
        vector_db_path=settings.llm.vector_database_path,
        embeddings_model_name=settings.llm.embeddings_model_name,
        provider_embeddings_name=settings.llm.provider_embeddings_name,
        _logger=logger,
    )
    # Chat with the bot
    chat_with_bot(
        vector_db=vector_db,
        embeddings_model_name=settings.llm.embeddings_model_name,
        provider_embeddings_name=settings.llm.provider_embeddings_name,
        student_level=student_level,
        level_relevant_chapters=level_relevant_chapters,
        course_level_infos=settings.course_levels,
        model_name=settings.llm.llm_model_name,
        provider_llm_name=settings.llm.provider_llm_name,
        prompt_path=prompt_path,
        logger=logger,
    )


if __name__ == "__main__":
    main()
