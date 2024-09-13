import os 
from typing import List, Tuple
from utils.load_cfg import LoadConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

APPCFG = LoadConfig()

class GOOGLE_MED:
    def respond(message:str) -> Tuple:
        # load vectordb
        vectorstore_disk = Chroma(
                        persist_directory=APPCFG.persist_directory,  # Directory of db
                        embedding_function=APPCFG.google_embedding   # Embedding model
                   )
        retriever = vectorstore_disk.as_retriever(search_kwargs={"k": APPCFG.top_k})

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


class LLM_MED:
    def __init__(self):
        self.tokenizer = APPCFG.tokenizer
        self.model = APPCFG.llm_model
        self.SYS_PROMPT = """You are an AI Medical Assistant trained on a vast dataset of health information. Please be thorough and
        provide an informative answer. If you don't know the answer to a specific medical inquiry, advise seeking professional help."""

    def rag_chatbot(self, prompt,k=2):
        scores , retrieved_documents = self.search(prompt, k)
        formatted_prompt = self.format_prompt(prompt,retrieved_documents,k)
        return self.generate(formatted_prompt)
    
    def format_prompt(self,prompt,retrieved_documents,k):
        """using the retrieved documents we will prompt the model to generate our responses"""
        PROMPT = f"Question:{prompt}\nContext:"
        for idx in range(k) :
            PROMPT+= f"{retrieved_documents['text'][idx]}\n"
        return PROMPT

    def generate(self,formatted_prompt):
        formatted_prompt = formatted_prompt[:2000] # to avoid GPU OOM
        messages = [{"role":"system","content":self.SYS_PROMPT},{"role":"user","content":formatted_prompt}]
        # tell the model to generate
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def search(self, query, k):
        """a function that embeds a new query and returns the most probable results"""
        embedded_query = self.ST.encode(query) # embed new query
        data = APPCFG.chroma_client.get_collection(name=APPCFG.collection_name)
        scores, retrieved_examples = data.get_nearest_examples( # retrieve results
            "embeddings", embedded_query, # compare our new embedded query with the dataset embeddings
            k=k # get only top k results
        )
        return scores, retrieved_examples 