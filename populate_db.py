import argparse, os, shutil, glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from tqdm import tqdm
import CodeLoader
from embedding import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader

CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
  # Check if the database should be cleared (using the --clear flag).
  parser = argparse.ArgumentParser()
  parser.add_argument("--reset", action="store_true", help="Reset the database.")
  args = parser.parse_args()
  if args.reset:
    print("Clearing Database")
    clear_database()

  documents = load_documents()
  add_to_chroma(documents)


def load_documents():
  files = [f for f in glob.iglob("./repos/**", recursive=True) if os.path.isfile(f)]
  documents = []
  for file in tqdm(files):
    if file.lower().endswith((".md", ".pdf", ".png")):
      continue
    try:
      document = CodeLoader.load(file)
      documents.append(document)
    except Exception as e:
      print(f"Error loading {file}: {e}")
  return documents

  # open every file in every folder

def add_to_chroma(chunks: list[Document]):
  # Load the existing database.
  db = Chroma(
    persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
  )

  db.add_documents(chunks)

def clear_database():
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
  main()
