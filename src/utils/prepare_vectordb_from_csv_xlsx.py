import os 
import pandas as pd 
from utils.load_cfg import LoadConfig
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

class PrepareVectorDBFromTabularData:
    """
    This class is designed to prepare a vector database from a CSV and XLSX file.
    It then loads the data into a ChromaDB collection. The process involves
    reading the CSV file, generating embeddings for the content, and storing 
    the data in the specified collection.
    
    Attributes:
        APPCFG: Configuration object containing settings and client instances for database and embedding generation.
        file_directory: Path to the CSV file that contains data to be uploaded.
    """
    def __init__(self,file_directory):
        """
        Initialize the instance with the file directory and load the app config.
        
        Args:
            file_directory (str): The directory path of the file to be processed.
        """
        self.APPCFG = LoadConfig()
        self.file_directory = file_directory

    def run_pipeline(self):
        """
        Execute the entire pipeline for preparing the database from the CSV.
        This includes loading the data, preparing the data for injection, injecting
        the data into ChromaDB, and validating the existence of the injected data.
        """
        self.df, self.file_name = self._load_dataframe(self.file_directory)
        self.docs= self._prepare_data_for_injection(df=self.df, file_name=self.file_name)
        self._inject_data_into_chromadb(self.docs)
        # self._validate_db

    def _load_dataframe(self, file_directory:str):
        """
        Load a DataFrame from the specified CSV or Excel file.
        
        Args:
            file_directory (str): The directory path of the file to be loaded.
            
        Returns:
            DataFrame, str: The loaded DataFrame and the file's base name without the extension.
            
        Raises:
            ValueError: If the file extension is neither CSV nor Excel.
        """
        file_name_with_extensions = os.path.basename(file_directory)
        file_name, file_extension = os.path.splitext(file_name_with_extensions)
        if file_extension == '.csv':
            data_df = pd.read_csv(file_directory)
        if file_extension == '.xlsx':
            data_df = pd.read_excel(file_directory)
        return data_df,file_name

    def _prepare_data_for_injection(self, df:pd.DataFrame, file_name:str):
        """
        Generate embeddings and prepare documents for data injection.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.
            file_name (str): The base name of the file  for use in metadata.
            
        Returns:
            Document of Langchain
        """
        docs = []
        
        ids = []
        
        for index, row in df.iterrows():
            output_str = ''
            for col in df.columns:
                output_str += f'{col}:{row[col]}\n'
            doc = Document(page_content= output_str, metadatas = {'source': file_name, 'ids': f'id{index}'})
            docs.append(doc)
        return docs

    def _inject_data_into_chromadb(self,data):
        """
        Inject the prepared data into ChromaDB.
        
        Raises an error if the collection_name already exists in ChromaDB.
        The method prints a confirmation message upon successful data injection.
        """
        vectorstore = Chroma.from_documents(
                     documents=data,                 # Data
                     embedding=self.APPCFG.google_embedding,    # Embedding model
                     persist_directory="/home/hungle/Project/RAG/data/chroma_db" # Directory to save data
                     )