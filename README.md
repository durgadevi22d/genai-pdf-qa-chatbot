## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
Manually searching through extensive PDF documents for specific information is inefficient and time-consuming. This project aims to develop a chatbot that can process PDF content, retrieve relevant information, and answer queries dynamically. By leveraging LangChain and vector-based retrieval methods, the solution provides accurate and efficient responses.

### DESIGN STEPS:

#### STEP 1:
Install required libraries: langchain, openai, chromadb, and pypdf. Obtain the OpenAI API key and store it securely using dotenv. Create a directory structure for your project with folders for documents and database storage.
#### STEP 2:
Load the PDF using PyPDFLoader to extract its content into pages. Split the content into manageable chunks using RecursiveCharacterTextSplitter. Embed the chunks using OpenAIEmbeddings and store them in a vector database using Chroma.
#### STEP 3:
Initialize a ChatOpenAI model for query processing. Set up a RetrievalQA chain using the vector database as a retriever. Query the chatbot and display the response based on the document's content.

### PROGRAM:
```
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']


loader = PyPDFLoader("docs/Flynn's classification.pdf")
pages = loader.load()

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)
docs = r_splitter.split_documents(pages)

embedding = OpenAIEmbeddings()
persist_directory = 'docs/chroma/'
metadata_docs = [Document(page_content=doc.page_content, metadata={"source": "docs/Flynn's classification.pdf"}) for doc in docs]
vectordb = Chroma.from_documents(
    documents=metadata_docs,
    embedding=embedding,
    persist_directory=persist_directory
)


vectordb.persist()


query = input("Enter your query related to flynn's classification document:")
llm_name = 'gpt-3.5-turbo'  
llm = ChatOpenAI(model_name=llm_name, temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)
result = qa_chain({"query": query})


print(result["result"])

```

### OUTPUT:
![exp 3 o](https://github.com/user-attachments/assets/fccd84ab-35fc-4943-b499-7ffe504c3bc9)


### RESULT:
Thus, the project successfully processed a PDF into vector embeddings with LangChain tools, implementing a chatbot via RetrievalQA to provide accurate, contextually relevant answers.
