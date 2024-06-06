Для запуску проекту потрібно:

1. Отримати токен OPENAI_API_KEY на https://platform.openai.com/api-keys та додати його до проекту.
2. Створити virtual enviroment: python -m venv venv
3. Активувати: .\venv\Scripts\activate
4. Встановити потрібні бібліотеки:
  pip install langchain streamlit langchain-openai beautifulsoup4 langchain-community chromadb PyMuPDF pypdf ragas datasets pandas faiss-cpu
5. Для запуску основної програми виконати: streamlit run app.py
6. Для запуску програми тестування виконати: streamlit run test_rag.py
