import argparse, github
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM as Ollama

from embedding import get_embedding_function

CHROMA_PATH = "chroma"
PROJECT = 'jabref/jabref'

PROMPT_TEMPLATE = """
You are an expert in the {project} project.
Given the following code explanation and GitHub issue tell us how would you solve an issue:

{context}

---

The issue you want to solve is the following one:
**Issue Title:** {title}
**Issue Body:** {body}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("issue_number", type=int, help="The issue_number.")
    args = parser.parse_args()
    issue_number = args.issue_number
    query_rag(issue_number)


def query_rag(issue_number: int):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    title, body = github.get_issue(PROJECT, issue_number)
    # Search the DB.
    results = db.similarity_search_with_score(f'**Issue Title:** {title}\n**Issue Body:** {body}', k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(project=PROJECT, context=context_text, title=title, body=body)

    model = Ollama(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
