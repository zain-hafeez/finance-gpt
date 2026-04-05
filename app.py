# app.py
"""
FinanceGPT — Gradio UI Entry Point
Module 8 (UI) + Module 9 (State Persistence)

M9 addition: each Gradio session gets a UUID as its session_id,
passed as LangGraph thread_id for isolated state persistence.
"""

import logging
import os
import time
import uuid

import gradio as gr

from src.data.loader import load_data
from src.data.validator import validate_file
from src.graph.nodes import clear_cache
from src.graph.workflow import run_query
from src.ui.components import (
    build_chart,
    build_chat_message,
    format_sql_display,
    format_status,
    format_table,
)
from src.utils.cache_setup import setup_cache
from src.utils.config import DB_PATH

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Initialize LangChain InMemoryCache (Layer 1 cache) ───────────────────────
setup_cache()


# ── Gradio event handlers ─────────────────────────────────────────────────────

def handle_file_upload(file):
    """
    Called when user uploads a file in the Gradio UI.
    Validates it, loads into SQLite, clears stale cache.
    Returns: (status_text, file_path, db_path)
    """
    if file is None:
        return "⬆️ Upload a file to get started.", None, None

    try:
        file_path = file if isinstance(file, str) else str(file)

        try:
            validate_file(file_path)
        except ValueError as e:
            return f"❌ File rejected: {e}", None, None

        result = load_data(file_path, DB_PATH)
        clear_cache()  # New file = stale cache — always clear

        filename = os.path.basename(file_path)
        status = (
            f"✅ **{filename}** loaded\n\n"
            f"📊 **{result['row_count']} rows** × "
            f"**{len(result['dataframe'].columns)} columns**\n\n"
            f"Columns: `{', '.join(result['columns'][:8])}`"
            + ("..." if len(result['columns']) > 8 else "")
        )
        return status, file_path, DB_PATH

    except Exception as e:
        logger.error("[upload] Unexpected error: %s", e, exc_info=True)
        return f"❌ Failed to load file: {e}", None, None


def handle_chat(user_message, chat_history, file_path, db_path, session_id):
    """
    Called when user sends a message in the chat.

    session_id is a gr.State — unique UUID per browser tab (set at UI init).
    This gets passed to run_query() as the LangGraph thread_id,
    so each tab has its own isolated, persisted conversation state.
    """
    if not file_path:
        updated_history = chat_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": "⬆️ Please upload a CSV or Excel file first."},
        ]
        return updated_history, None, None, "⬆️ No file loaded", "_No query_"

    # Use the session's UUID as LangGraph thread_id
    # If for any reason session_id is None, fall back to "default"
    effective_session_id = session_id if session_id else "default"

    t_start = time.time()
    result = run_query(
        question=user_message,
        file_path=file_path,
        db_path=db_path,
        session_id=effective_session_id,   # ← M9: session isolation
    )
    elapsed = time.time() - t_start

    bot_reply = build_chat_message(result)

    updated_history = chat_history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": bot_reply},
    ]

    return (
        updated_history,
        build_chart(result.get("raw_result"), result.get("query_type", ""), user_message),
        format_table(result.get("raw_result")),
        format_status(result, elapsed),
        format_sql_display(result),
    )


# ── Build the Gradio UI ───────────────────────────────────────────────────────

def build_ui():
    """
    Constructs the Gradio Blocks UI.

    gr.State components are invisible — they hold data between events:
      - file_path_state: path to the uploaded file
      - db_path_state:   path to the SQLite db
      - session_id_state: unique UUID per tab (M9)
    """
    with gr.Blocks(title="FinanceGPT") as demo:

        gr.Markdown(
            "# 💰 FinanceGPT\n"
            "Upload a CSV or Excel file, then ask questions in plain English."
        )

        # ── Invisible state holders ───────────────────────────────────────
        file_path_state  = gr.State(value=None)
        db_path_state    = gr.State(value=None)

        # M9: session_id is set to a new UUID at page load.
        # lambda: str(uuid.uuid4()) is called once per Gradio session (per tab).
        # This UUID becomes the LangGraph thread_id — uniquely identifies this user's state.
        session_id_state = gr.State(value=lambda: str(uuid.uuid4()))

        # ── Layout ────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload Data")
                file_input    = gr.File(
                    label="Upload CSV or Excel",
                    file_types=[".csv", ".xlsx"],
                )
                upload_status = gr.Markdown("⬆️ Upload a file to get started.")

            with gr.Column(scale=2):
                gr.Markdown("### 💬 Ask Questions")
                chatbot = gr.Chatbot(
                    label="FinanceGPT",
                    height=400,
                       # Gradio 6.x format
                )
                with gr.Row():
                    chat_input  = gr.Textbox(
                        placeholder="e.g. What is the total revenue by region?",
                        label="Your question",
                        scale=4,
                    )
                    submit_btn  = gr.Button("Ask", variant="primary", scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Chart")
                chart_output  = gr.Plot(label="Visualization")
            with gr.Column():
                gr.Markdown("### 📋 Raw Results")
                table_output  = gr.Dataframe(label="Query Results")

        with gr.Row():
            status_output = gr.Markdown("_Ready_")
            sql_output    = gr.Markdown("_No query yet_")

        # ── Event wiring ──────────────────────────────────────────────────
        file_input.change(
            fn=handle_file_upload,
            inputs=[file_input],
            outputs=[upload_status, file_path_state, db_path_state],
        )

        # Both the button click and pressing Enter on the textbox trigger handle_chat
        for trigger in [submit_btn.click, chat_input.submit]:
            trigger(
                fn=handle_chat,
                inputs=[
                    chat_input,
                    chatbot,
                    file_path_state,
                    db_path_state,
                    session_id_state,   # ← M9: pass session ID to handler
                ],
                outputs=[
                    chatbot,
                    chart_output,
                    table_output,
                    status_output,
                    sql_output,
                ],
            ).then(
                fn=lambda: "",        # Clear the input box after submit
                outputs=[chat_input],
            )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )