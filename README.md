# 🔥 LangGraph Chatbot with Groq & Gemma 2

A blazing-fast, stateful chatbot built using [LangGraph](https://docs.langchain.com/langgraph/), [Groq](https://groq.com/), and [Google's Gemma 2 9B It](https://ai.google.dev/gemma) model.  
Powered by modular graph logic, lightning-speed inference, and real-time conversation tracing.

## 🚀 Features

- 🧠 Uses **Gemma 2 9B It** via **Groq API** for high-performance LLM inference  
- 🔄 Built with **LangGraph** for graph-based conversation flow  
- 🔍 Integrated with **LangSmith** for seamless tracing/debugging  
- 💬 Maintains message history using `add_messages`  
- 📈 Visualizes the LangGraph as a Mermaid diagram

## 📦 Requirements

- Python 3.10+
- Groq API key
- LangSmith API key
- Google Colab (recommended for quickstart)

## 🔧 Setup
🧠 Define the Graph
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

class State(TypedDict):
  messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return { "messages": llm.invoke(state["messages"]) }

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
##💬 Chat Loop
graph = graph_builder.compile()

while True:
    user_input = input("user: ")
    if user_input.lower() in ["quit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({ "messages": [("user", user_input)] }):
        for values in event.values():
            print(values["messages"])
##🖼️ Visualize the Graph
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))




