import os
from dotenv import load_dotenv
import json

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage

from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

llm = ChatOpenAI(model = "gpt-3.5-turbo", temperature = 0)

tool = TavilySearchResults(max_result = 2)
tools = [tool]

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# next we need to create a function to actually run the tools if they are called. 
# we will do this by adding the tools to a new node.

class BasicToolNode:
    """A node that runs the tools requested in the last AI message"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
    
    def __call__(self, inputs: dict):
        if messages:= inputs.get("messages", []):
            messsage = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []

        for tool_call in messsage.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content = json.dumps(tool_result),
                    name = tool_call["name"],
                    tool_call_id = tool_call["id"],
                )
            )
        
        return {"messages": outputs}

tool_node = BasicToolNode(tools = [tool])
graph_builder.add_node("tools", tool_node)

# __start__ -> chatbot -> __end__
#                ^
#                |--> tools

# With the tool node added, we can define the conditional_edges.

# Recall that edges route the control flow from one node to the next. Conditional edges usually contain "if" statements to route to different nodes depending on the current graph state. These functions receive the current graph state and return a string or list of strings indicating which node(s) to call next.

# Below, call define a router function called route_tools, that checks for tool_calls in the chatbot's output. Provide this function to the graph by calling add_conditional_edges, which tells the graph that whenever the chatbot node completes to check this function to see where to go next.

# The condition will route to tools if tool calls are present and END if not.

def route_tools(state: State):
    """
    Use inthe conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages:= state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    
    return END

# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

from IPython.display import Image
import os

# Generate the Mermaid graph image
mermaid_png = graph.get_graph().draw_mermaid_png()

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the file path for the image in the same directory as the script
file_path = os.path.join(script_dir, "mermaid_graph.png")

# Save the image
with open(file_path, "wb") as f:
    f.write(mermaid_png)

print(f"Mermaid graph image saved as: {file_path}")
