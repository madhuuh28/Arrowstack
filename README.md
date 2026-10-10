RAG Knowledge Assistant — Sample Documents

Files:
- documents/leave_policy.pdf
- documents/work_from_home_policy.pdf
- documents/code_of_conduct.pdf
- evaluation_questions.txt

All policies are fictional and intended only for educational RAG testing. They do not represent a real company or provide legal/HR advice.

Setup your Python project:
1. Place rag_assistant.py beside the documents folder.
2. Install: python -m pip install -U google-genai pypdf numpy
3. Set GEMINI_API_KEY in your terminal environment.
4. Run: python rag_assistant.py

Ask questions from the PDFs and compare answers with evaluation_questions.txt.
