import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics  import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from dotenv import load_dotenv
import json

load_dotenv()

def main():

    st.set_page_config(page_title="Test LLM")

    st.header("Testing BotFin")

    pdf = st.file_uploader('Upload your PDF', type="pdf")

    text= ""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_spliter = RecursiveCharacterTextSplitter()

    chunks = text_spliter.split_text(text)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)


    if st.button("Process"):
        with st.spinner("Processing"):
            retriever = vectorstore.as_retriever()

            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

            template = """
                You are a helpful, respectful and honest assistant for question-answering tasks. 
                Always answer as helpfully as possible, while being safe.
                Your answer should not include any harmful, unethical, racist, sexist, toxic,
                dangerous, or illegal content. Please ensure that tour responses are socially 
                unbiased and positive in nature.
                If a quesyion does not make sense, or is not factually coherent,
                explain why instead of answering something not correct.
                If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                Question: {question}
                Context: {context}
                Answer: 
            """

            prompt = ChatPromptTemplate.from_template(template)

            rag_chain = (
                {"context": retriever, "question":RunnablePassthrough()}
                | prompt
                |llm
                |StrOutputParser()
            )

            questions, ground_truths = [], []

            with open("dataset.json") as json_file:
                data = json.load(json_file)
                for item in data:
                    questions.append(item['question'])
                    ground_truths.append(item['answer'])
        
            answers, contexts = [], []

            for query in questions:
                answers.append(rag_chain.invoke(query))
                contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

            data = {
                "question": questions,
                "answer"  : answers,
                "contexts": contexts,
                "ground_truth" :  ground_truths 
            }

            dataset = Dataset.from_dict(data)
            result = evaluate(
                dataset=dataset, 
                metrics=[
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                ]
            )

            df = result.to_pandas()

            st.write(df)

if __name__ == "__main__":
    main()