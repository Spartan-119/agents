from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Define tools for the agent to use
@tool
def search(query: str):
    """This method calls to surf the web."""
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]
tool_node = ToolNode(tools)

# Initialize the model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# define the function that determines whether to continue or not..
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]

    # if the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # otherwise we stop (we reply to the user)
    return END

# define the function that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # we return a list, because this will get added to the existing list
    return {"messages": [response]}

# define a new graph
workflow = StateGraph(MessagesState)

# define teh two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# set the entrypoint as 'agent'
# this means that this node is the first one called
workflow.add_edge(START, "agent")

# we now add a conditional edge
workflow.add_conditional_edges(
    # first, we define the start node. we use "agent".
    # this means these are the edges taken after the "agent" node is called.
    "agent",
    # next, we pass in the function that will determine which node is called next.
    should_continue,
)

# we now add a normal edge from "tools" to "agent".
# thsi means that after "tools" is called, "agent" node is called next.
workflow.add_edge("tools", "agent")

# initialise memory to persist state between graph runs
checkpointer = MemorySaver()

# finally, we compile it!
# thsi compiles it into a langchain runnable.
# meaning you can use it as you would any other runnable.
# note that we are (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer = checkpointer)

# use the runnable
final_state = app.invoke(
    {
        "messages": [HumanMessage(content = "what is the weather in sf")]
    },
    config = {"configurable": {"thread_id": 42}}
)
print(final_state["messages"][-1].content)