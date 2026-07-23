from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="documents")
documents = reader.load_data()