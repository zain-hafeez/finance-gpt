# app.py
# ------------------------------------------------------------
# FinanceGPT — Main Application Entry Point
# ------------------------------------------------------------
# This is the file you run to start the app: python app.py
#
# MODULE 0: Hello World version — proves the environment works.
# This file will be fully replaced in Module 8 with the real
# Gradio UI wired to the LangGraph workflow.
# ------------------------------------------------------------

import gradio as gr

# A simple function that the UI will call when the user types something.
# In the real app, this will invoke the full LangGraph workflow.
def greet(name: str) -> str:
    """Placeholder function — will be replaced in Module 8."""
    if not name:
        return "Please enter your name!"
    return f"Hello, {name}! FinanceGPT environment is working correctly."

# Build the Gradio interface.
# gr.Interface is the simplest way to create a UI:
# - fn: the Python function to call when the user submits input
# - inputs: what the user sees for input (a text box here)
# - outputs: what the user sees for output (a text box here)
with gr.Blocks(title="FinanceGPT") as demo:
    gr.Markdown("# 🤖 FinanceGPT")
    gr.Markdown("**Module 0 — Environment Check**  \nIf you can see this, your setup is working correctly.")
    
    with gr.Row():
        name_input = gr.Textbox(
            label="Enter your name to test",
            placeholder="e.g. Ahmed"
        )
    
    submit_btn = gr.Button("Test Environment", variant="primary")
    output_text = gr.Textbox(label="Result", interactive=False)
    
    submit_btn.click(fn=greet, inputs=name_input, outputs=output_text)

# Launch the app.
# share=False means it only runs on your local machine (not publicly).
# The app will be accessible at http://127.0.0.1:7860 in your browser.
if __name__ == "__main__":
    demo.launch(share=False)
    
