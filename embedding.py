from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
  return OllamaEmbeddings(model="llama3.1:8b")