from dotenv import load_dotenv
from uuid import uuid4
from pathlib import Path

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()



#Constants
output_parser= StrOutputParser()
chunck_size= 1000
vector_store_dir= Path(__file__).parent/"resources/vectorstore"
embedding_model= 'sentence-transformers/all-MiniLM-L6-v2'
collection_name= "real_estate"
prompt= PromptTemplate(
    template="""You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
        """,
    input_variables=['context','question']
)


llm= None
embedding= None
vector_store= None


def initialize_components():
    global llm, embedding,vector_store
    print(" >> Initializing the llm")
    if llm == None:
        llm= ChatOpenAI()

    print(" >> Initializing the embedding model") 
    if embedding == None:
        embedding= HuggingFaceEmbeddings(model_name=embedding_model)
    
    
    print(" >> Initializing the vector store")
    if vector_store== None:
        vector_store= Chroma(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=vector_store_dir
        )


def process_urls(urls):
    """
    Docstring for process_urls
    
    :param urls: input urls
    :return : index documents
    """
    yield"Initializing components: "
    initialize_components()

    yield "Erasing existing storage in the Vector store"
    vector_store.reset_collection()

    yield "Loading Documents"
    loader= WebBaseLoader(urls)
    data= loader.load()
    # print(f"Loaded {len(data)} documents. First 100 characters: {data[0].page_content[:100]}")
    
    yield "Splitting text"
    text_splitter= RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',' '],
        chunk_size= chunck_size
    )
    docs= text_splitter.split_documents(data)

    uuids= [str(uuid4()) for i in range(len(docs))]

    print("Adding documents to the vector store")
    vector_store.add_documents(docs,ids=uuids)


def format_context(docs):
    context= '\n\n'.join(doc.page_content for doc in docs)
    return context

def generate_answers(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    #yield "Generating answer for the given query"

    #yield">> Initializing the retriever"
    retriever= vector_store.as_retriever(search_type='similarity', search_kwargs={'k':2})
    #yield">> Fetching relevent documents using retriever"
    docs= vector_store.similarity_search(query=query,k=4)
    # yield">> Joining text using format_context function"
    context = format_context(docs)
    # print("--- DEBUG: CONTEXT SENT TO LLM ---")
    # print(context) 
    # print("---------------------------------")

    parallel_chain= RunnableParallel(
        {'question': RunnablePassthrough(),
         'context': retriever | RunnableLambda(format_context)
        }
    )
    # yield">> Initializing the final chain"
    final_chain= parallel_chain | prompt | llm | output_parser
    # yield">> Invoking the final chain with query"
    response= final_chain.invoke(query)
    # source= response.get("sources", "")
    yield  response



if __name__ == "__main__":
    urls = ['https://blog.fabric.microsoft.com/en-US/blog/announcing-the-winners-of-hack-together-the-microsoft-data-ai-kenya-hack/']
    process_urls(urls)
    
    print("\n--- Generating Result ---")
    query = "What is The Microsoft Data + AI Kenya Hack"
    
    # Loop through the generator to see status and get the final result
    result = None
    for step in generate_answers(query):
        print(step)  # This shows your ">>" status updates in the console
        result = step # The very last thing yielded will be stored here
    
    print(f'\nFinal Answer: {result}')

