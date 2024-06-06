import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import fitz

load_dotenv()

def merge_pdfs(pdf_list):
    merged_pdf = fitz.open()  
    for pdf in pdf_list:
        pdf_document = fitz.open(pdf)
        merged_pdf.insert_pdf(pdf_document)
    return merged_pdf

def extract_text_from_pdf(merged_pdf):
    pdf_document = merged_pdf
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def get_text_chunks(pdf_list):
    merged_pdf = merge_pdfs(pdf_list)
    text =  extract_text_from_pdf(merged_pdf)
    text_splitter = RecursiveCharacterTextSplitter()
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_document_chunks(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    return document_chunks

def get_vectorstore_from_pdf(text_chunks):
    vector_store = Chroma.from_texts(text_chunks, OpenAIEmbeddings())
    return vector_store

def get_vectorstore_from_url(document_chunks):
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

def merge_vector_stores(source_vector_stores, destination_vector_store):
    for source_store in source_vector_stores:
        all_ids = source_store.all_ids()
        for vector_id in all_ids:
            vector_data = source_store.get(vector_id)
            destination_vector_store.add(vector_data)

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", '''
            You are a helpful, respectful and honest assistant for question-answering tasks. 
            Always answer as helpfully as possible, while being safe.
            Your answer should not include any harmful, unethical, racist, sexist, toxic,
            dangerous, or illegal content. Please ensure that tour responses are socially unbiased and positive in nature.
            If a question does not make sense, or is not factually coherent,
            explain why instead of answering something not correct.
            If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Given the above conversation, generate a search query to look up in order to get information relevant to the conversation
            '''
         )
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the below context to answer the user's questions: \n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    
    conversation_rag_chain = get_conversational_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })

    return response['answer']

def main():

    st.set_page_config(page_title="BotFin")

    st.title("Your financial helper!")

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("Resources")
        website_url = st.text_input("Website URL",)
        
        pdf_docs = st.file_uploader("Upload PDFs ", accept_multiple_files=True)
  
        if st.button("Process"):
            with st.spinner("Processing"):
                st.session_state.chat_history = []
                
                if not (website_url is None or website_url == "") and len(pdf_docs):
                    text_chunks = get_text_chunks(pdf_docs)
                    document_chunks = get_document_chunks(website_url)
                    vector_store = get_vectorstore_from_pdf(text_chunks)
                    vector_store.add_documents(document_chunks)    
                    st.session_state.vector_store = vector_store
                elif len(pdf_docs):
                    print(2)
                    text_chunks = get_text_chunks(pdf_docs)
                    st.session_state.vector_store = get_vectorstore_from_pdf(text_chunks)
                elif not website_url is None or website_url != "":
                    print(3)
                    document_chunks = get_document_chunks(website_url)
                    st.session_state.vector_store = get_vectorstore_from_url(document_chunks)
                

    user_query = st.chat_input("Type your query...")
    if user_query != "" and user_query is not None:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        

    for msg in st.session_state.chat_history:
        if isinstance(msg, AIMessage):
            with st.chat_message("AI"):
                st.write(msg.content)
        elif isinstance(msg, HumanMessage):
            with st.chat_message("Human"):
                st.write(msg.content)

if __name__ == "__main__":
    main()