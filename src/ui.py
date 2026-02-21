"""
Legacy Streamlit UI has been removed.

Run the chat-style web app instead:

  uv run fileloop
  # or
  uvicorn src.app:app --reload --host 0.0.0.0 --port 8501

Then open http://localhost:8501 in your browser.
"""


def main():
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8501)
