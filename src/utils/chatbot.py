import os 
from typing import List, Tuple
from utils.load_cfg import LoadConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
APPCFG = LoadConfig()

class ChatBot:
    """
    A ChatBot class capable of responding to messages using different modes of operation.
    It can interact with CSV file, leverage language chain agents for Q&A,
    and use embeddings for Retrieval-Augmented Generation (RAG) with ChromaDB.
    """
    def respond(message:str) -> Tuple:
        """
        Respond to a message based on the given chat and application functionality types.

        Args:
            chatbot (List): A list representing the chatbot's conversation history.
            message (str): The user's input message to the chatbot.
            chat_type (str): Describes the type of the chat (interaction with Vector DB or RAG).
            app_functionality (str): Identifies the functionality for which the chatbot is being used (e.g., 'Chat').

        Returns:
            Tuple[str, List, Optional[Any]]: A tuple containing an empty string, the updated chatbot conversation list,
                                             and an optional 'None' value. The empty string and 'None' are placeholder
                                             values to match the required return type and may be updated for further functionality.
                                             Currently, the function primarily updates the chatbot conversation list.
        """
        # load vectordb
        vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",       # Directory of db
                        embedding_function=APPCFG.google_embedding   # Embedding model
                   )
        retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 1})

        llm = APPCFG.google_llm

        llm_prompt_template = """You are an assistant for question answering tasks.
        Your task is to give a conclusion about the relevant disease based on the following context provided by the user.
        If you don't know the answer, just say you don't know.
        Use a maximum of five sentences and keep your answer short.\n
        Question: {question} \nContext: {context} \nAnswer:"""
    
        llm_prompt = PromptTemplate.from_template(llm_prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | llm_prompt
        | llm
        | StrOutputParser()
        )

        return rag_chain.invoke(message)
