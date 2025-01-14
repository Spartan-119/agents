import os
from dotenv import load_dotenv

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

class State(TypedDict):
    # messages have the type "list". The "add_messages" function 
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them.)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# the first argument is the unique node name
# the second argument is the function or the object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# next, we add an entry point that tells 
# our graph where to start its work each time we run it.
graph_builder.add_edge(START, "chatbot")

# similarly we set a finish point that instructs the 
# graph "anytime this node is run, you can exit"
graph_builder.add_edge("chatbot", END)

# Finally, we'll want to able to run our graph.
# for that, we use compile() on the graph builder.
# this creates a CompiledGraph we can use invoke on our state.
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# from IPython.display import Image
# import os

# # Generate the Mermaid graph image
# mermaid_png = graph.get_graph().draw_mermaid_png()

# # Get the directory of the current script
# script_dir = os.path.dirname(os.path.abspath(__file__))

# # Create the file path for the image in the same directory as the script
# file_path = os.path.join(script_dir, "mermaid_graph.png")

# # Save the image
# with open(file_path, "wb") as f:
#     f.write(mermaid_png)

# print(f"Mermaid graph image saved as: {file_path}")


# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break

#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "What do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break

#####################################################3
# enhancing the chatbot with tools

tool = TavilySearchResults(max_result = 2)
tools = [tool]
response = tool.invoke("What is a 'node' in LangGraph?")
print(response)