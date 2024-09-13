import os 
from dotenv import load_dotenv
import yaml
import chromadb
import torch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import transformers
from transformers import BitsAndBytesConfig

class LoadConfig:
    def __init__(self, use_google_api = False):
        with open('../configs/app_config.yml') as cfg:
            app_config = yaml.load(cfg, Loader=yaml.FullLoader)
        
        self.load_directories(app_config=app_config)
        # using google gemini
        if use_google_api == True:
            self.load_googleai_model(app_config)
        # using LLM model 
        if use_google_api == False:
            self.load_llm_model(app_config=app_config)

        self.load_rag_configs(app_config=app_config)
        self.load_chroma_client()

    def load_directories(self,app_config):
        self.stored_csv_xlsx_directories = app_config['directories']['stored_csv_xlsx_directory']
        self.persist_directory = app_config['directories']['persist_directory']

    def load_llm_model(self,app_config):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(app_config['llm_configs']['model_pretrain'])
        self.llm_model = transformers.LlamaForCausalLM.from_pretrained(app_config['llm_configs']['model_pretrain'],
                                                                    torch_dtype=torch.bfloat16,
                                                                    device_map="auto",
                                                                    quantization_config=bnb_config)

    def load_googleai_model(self,app_config):
        os.environ['GOOGLE_API_KEY'] = app_config['google_ai_config']['GOOGLE_API_KEY']
        # load google embedding 
        self.google_tokenizer = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
        # load gemini 1.5 pro 
        self.google_llm = ChatGoogleGenerativeAI(
            model = 'gemini-1.5-pro',
            temperature = 0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        
    def load_chroma_client(self):
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
    
    def load_rag_configs(self,app_config):
        self.collection_name = app_config['rag_config']['collection_name']
        self.top_k = app_config['rag_config']['top_k']