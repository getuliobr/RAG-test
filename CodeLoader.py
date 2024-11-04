from langchain_core.documents import Document
from langchain_ollama import OllamaLLM as Ollama
from langchain.prompts import ChatPromptTemplate

PROMPT_TEMPLATE = """
Based on the code and path describe what you think the following code does in plain English:
PATH: {code_path}
CODE: ```
{code}
```
"""
model = Ollama(model="llama3.1:8b")

def load(path):
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  with open(path, "r") as f:
    code = f.read()
  prompt = prompt_template.format(code_path=path, code=code)
  response_text = model.invoke(prompt)
  return Document(response_text, metadata={"source": path})