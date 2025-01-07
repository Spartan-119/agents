from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

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


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break