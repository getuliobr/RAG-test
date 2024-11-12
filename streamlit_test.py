import streamlit as st
from langchain_ollama import ChatOllama as Ollama

st.title("Simple chat")

model = Ollama(model="llama3.1:8b")

# Initialize chat history
if "messages" not in st.session_state:
  st.session_state.messages = [{"role": "system", "content": "You are an expert programmer."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
  # Display user message in chat message container
  with st.chat_message("user"):
      st.markdown(prompt)
  # Add user message to chat history
  st.session_state.messages.append({"role": "user", "content": prompt})

  # Display assistant response in chat message container
  with st.chat_message("assistant"):
    response = st.write_stream(model.stream(st.session_state.messages))
  # Add assistant response to chat history
  st.session_state.messages.append({"role": "assistant", "content": response})