"""Streamlit entry point: ``streamlit run streamlit_app.py``.

Streamlit needs a script path, not a module, so this file exists purely to be that
path. It stays a one-liner: the app is a package module like everything else, and a
top-level script that accumulated logic would be a second place for the project's code
to live.
"""

from rag_eval.ui.app import main

main()
