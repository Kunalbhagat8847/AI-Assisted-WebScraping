from dotenv import load_dotenv
import os

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Conversation imports
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

embeddings_dir = os.path.join("embeddings")
os.makedirs(embeddings_dir, exist_ok=True)
api_key="Enter api key"


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def scrape_website(url):
    # response = requests.get(url)
    response = requests.get(url, timeout=10)  # Timeout set to 10 seconds
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ""
    for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'ul', 'ol']):
        text += tag.get_text(strip=True) + " "

    # Preprocess the text
    text = preprocess_text(text)
    # Store the text into a file
    file_path = os.path.join(os.getcwd(), "scraped_text.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text)
    return file_path,file

def create_embeddings_and_save(file, embeddings_dir, api_key):
    file_name = os.path.basename(file)
    embedding_filename = os.path.splitext(file_name)[0] + ".pkl"
    embedding_path = os.path.join(embeddings_dir, embedding_filename)
    if os.path.exists(embedding_path):
        print(f"Embeddings already present for {file_name}: {embedding_filename}")
        return embedding_filename
    else:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()  # Read the content of the text file
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = splitter.create_documents([text])  # Use the text here instead of pdf_text

        embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        vector_index = FAISS.from_documents(text_chunks, embeddings)
        serialized_db = vector_index.serialize_to_bytes()
        with open(embedding_path, "wb") as f:
            f.write(serialized_db)
        print(f"Created embeddings for `{file_name}` and saved it to `{embedding_path}`")
        return embedding_filename
def load_embeddings(file):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    with open(file, "rb") as f:
        pkl = f.read()

    vectorStore = FAISS.deserialize_from_bytes(
        serialized=pkl,
        embeddings=embeddings
    )
    print("Embeddings loaded Now you can chat with your Document")
    return vectorStore


def trim_chat_history(chat_history):
    return chat_history[-10:]


def create_chain(vectorStore):
    model = ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        temperature=0.2
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the question below only from the text provided. Answer in detail and in a friendly, enthusiastic tone. If related data doesn't lies in the context, don't try to make answers by your own: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retrieval_chain = create_retrieval_chain(

        history_aware_retriever,
        chain
    )

    return retrieval_chain


def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "chat_history": trim_chat_history(chat_history),
        "input": question+"in provided text",
    })

    return response["answer"]

if __name__ == '__main__':
    # Scrape website
    url = input("Enter the URL: ") # Change to your desired URL
    file_path, file_name = scrape_website(url)

    # Create embeddings from scraped text and save them
    embedding_filename = create_embeddings_and_save(file_path, embeddings_dir, api_key)

    # Load embeddings
    db = load_embeddings(os.path.join(embeddings_dir, embedding_filename))

    # Create conversation chain
    chain = create_chain(db)

    # Initialize chat history
    chat_history = []

    while True:
        user_input = input("\nYou : ")
        if user_input.lower() == "exit":
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant : ", response)
