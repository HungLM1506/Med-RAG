import os 
from dotenv import load_dotenv
import yaml
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

class LoadConfig:
    def __init__(self) -> None:
        with open('/home/hungle/Project/RAG/configs/app_config.yml') as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        
        self.load_directories(app_config=app_config)
        self.load_llm_configs(app_config=app_config)
        self.load_rag_configs(app_config=app_config)
        self.load_chroma_client()
        self.load_googleai_model()

    def load_directories(self,app_config):
        self.stored_csv_xlsx_directories = app_config['directories']['stored_csv_xlsx_directory']
        self.persist_directory = app_config['directories']['persist_directory']

    def load_llm_configs(self,app_config):
        pass

    def load_googleai_model(self):
        os.environ['GOOGLE_API_KEY'] = 'AIzaSyDzYEtOw-rxGSj_bKcR1QkSW9NeWL0k_UA'
        self.google_llm = ChatGoogleGenerativeAI(
            model = 'gemini-1.5-pro',
            temperature = 0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.google_embedding = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        
    def load_chroma_client(self):
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
    
    def load_rag_configs(self,app_config):
        self.collection_name = app_config['rag_config']['collection_name']
        self.top_k = app_config['rag_config']['top_k']