from utils.prepare_vectordb_from_csv_xlsx import PrepareVectorDBFromTabularData

if __name__=="__main__":
    from pyprojroot import here
    # Specify the path to your CSV file directory below
    medical_data = '/home/hungle/Project/RAG/data/test_data/diabetes.csv'
    # Create an instance of the PrepareVectorDBFromTabularData class with the file directory
    data_prep_instance = PrepareVectorDBFromTabularData(file_directory=medical_data)
    # Run the pipeline to prepare and inject the data into the vector database
    data_prep_instance.run_pipeline()