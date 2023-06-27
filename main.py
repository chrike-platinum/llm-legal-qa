import streamlit as st
import openai
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import random

random.seed(42)  


# Set up credentials
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_BASE = st.secrets["AZURE_OPENAI_API_BASE"]
AZURE_OPENAI_API_TYPE = 'azure'
AZURE_OPENAI_API_VERSION = '2023-03-15-preview'
AZURE_OPENAI_EMBEDDINGS_MODEL_NAME= 'accounton-gpt-ada-text-embedder'
AZURE_OPENAI_GPT4_MODEL_NAME= 'accounton-gpt-35-turbo'
PINECONE_INDEX_NAME='qa-legal-case'
PINECONE_API_KEY=st.secrets["PINECONE_API_KEY"]

default_prompt_template="""

You are Belgian lawyer. Conduct legal research on for the following question (QUESTION) given the the case context (CASE CONTEXT). Please be concise. Summarize the relevant case law, statutes, and regulations.
For your legal research you can make use of the extracted parts of long documents.
Provide analysis and conclusions based on your research. 
given the following extracted parts of a long document and a question. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer that includes the list of sources that you used.

Provide your answer always in Dutch and structure the answer as how a lawyer would do it.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Wij verdedigen partij Z. Bouw een argumentatie op om partij Z te verdedigen. Maak een conclusie op, op basis van de geldende rechtspraak en wetgeving.

CASE CONTEXT: {case_context}

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

case_context_info="""
1.Partij X (de principaal) en partij Y (de handelsagent) – beide ondernemingen – sluiten op 18 juli 2011 een handelsagentuurovereenkomst.

De essentiële bepalingen van deze overeenkomst zijn de volgende:
-Het territorium is het Verenigd Koninkrijk en Ierland.
-De agentuur is exclusief, in de zin dat geen andere agent voor hetzelfde territorium mag worden aangesteld.
-De commissie bedraagt 15% .
-De overeenkomst is gesloten voor onbepaalde duur vanaf 18 juli 2011.
-Het recht op een uitwinningsvergoeding wordt beperkt tot 1 maand.
-De overeenkomst valt onder Belgisch recht.

2.Partij Y gaat vrijwillig in vereffening en de handelsagentuurovereenkomst wordt verdergezet door partij Z (met dezelfde bestuurder als partij Y). Partijen sluiten geen nieuwe overeenkomst, waarin de contractsoverdracht wordt geformaliseerd, maar partij Z zet de activiteiten wel verder. De nieuwe commissiefacturen worden voortaan eenvoudigweg vanuit partij Z verstuurd en door partij Y in tempore non suspecto betaald.


3. Op 1 juni 2018 beëindigt partij X de handelsagentuurovereenkomst eenzijdig, zonder een opzeggings- of andere vergoeding te betalen. Zij stelt geen overeenkomst te hebben met partij Z. Partij Z heeft haar voorgelogen en is een volstrekte derde.


Partij Z vordert een opzeggingsvergoeding en een uitwinningsvergoeding op basis van de handelsagentuurovereenkomst die Partijen X en Y op 18 juli 2011 hebben gesloten en trekt naar de rechtbank."""

default_question="Wie draagt de bewijslast voor de uitwinningsvergoeding?"



embedder = OpenAIEmbeddings(
    openai_api_base=AZURE_OPENAI_API_BASE, 
    openai_api_key=AZURE_OPENAI_API_KEY, 
    openai_api_type=AZURE_OPENAI_API_TYPE,
    deployment=AZURE_OPENAI_EMBEDDINGS_MODEL_NAME,
    model='text-embedding-ada-002',
    chunk_size=1)


llm = AzureChatOpenAI(
    temperature=0,
    #top_p=0.0001,
    openai_api_base=AZURE_OPENAI_API_BASE, 
    openai_api_key=AZURE_OPENAI_API_KEY, 
    openai_api_version=AZURE_OPENAI_API_VERSION, 
    deployment_name=AZURE_OPENAI_GPT4_MODEL_NAME)



pinecone.init(
    api_key=PINECONE_API_KEY,
    environment="eu-west4-gcp")


vectordb = Pinecone(
    index=pinecone.Index(PINECONE_INDEX_NAME),
    embedding_function=embedder.embed_query,
    text_key="text"
)



# Streamlit app
def main():
    # Logo
    logo_image = "https://static.wixstatic.com/media/0a5b13_c07752f7a8324e568649f0341d2ff5e5~mv2.png"  # Replace with the path to the logo image
    st.image(logo_image, use_column_width=True)
    
    st.title("Legal LLM assistant Model")
    

    # LLM Prompt input
    st.subheader("LLM Prompt")
    prompt = st.text_area("Enter your LLM prompt here:", height=350,value=default_prompt_template)

    # Context input
    st.subheader("Case context Information")
    context = st.text_area("Enter your context information here:", height=350, value=case_context_info)

    # Questions input
    st.subheader("Questions")
    questions = st.text_area("Enter your questions here:", height=350,value=default_question)

    # Generate responses
    if st.button("Generate Responses"):
        if prompt and context and questions:
            # Split questions into a list
            question_list = questions.split("\n")

            # Generate responses for each question
            for question in question_list:
                st.write(f"Question: {question}")
                response = generate_response(prompt, context, question,vectordb)
                st.write(f"Response: {response['answer']}")
                st.write(f"sources: {response['sources']}")


# Generate response using OpenAI API
def generate_response(prompt_template, context, question,vectordb):
    #query_vector=create_embedding()
    #relevant_documents = pinecone_index.query(vector=query_vector)
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question","case_context"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    print('retriever created')
    qa_stuff = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, 
    chain_type="stuff",#"stuff", 
    retriever=vectordb.as_retriever(), 
    verbose=True,
    chain_type_kwargs=chain_type_kwargs
    )
    answer=qa_stuff({"question": question,"case_context":context})
    print(answer)
    return answer


if __name__ == "__main__":
    main()
