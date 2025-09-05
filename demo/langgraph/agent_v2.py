# env OTEL_PYTHON_DISTRO=aws_distro \
#            OTEL_PYTHON_CONFIGURATOR=aws_configurator \
#            OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
#            OTEL_EXPORTER_OTLP_LOGS_HEADERS="x-aws-log-group=test/genesis,x-aws-log-stream=default,x-aws-metric-namespace=genesis" \
#            OTEL_RESOURCE_ATTRIBUTES="service.name=langchain-instrumentor-bedrock-test-app" \
#            AGENT_OBSERVABILITY_ENABLED="true" \
#            opentelemetry-instrument python  "/Users/ehrican/Desktop/langgraph sample/agent_v2.py"

from typing import Annotated
import os
from typing import List, Dict, Any, TypedDict
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import boto3
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chat_models import init_chat_model
# from langchain.agents import AgentExecutor, create_react_agent
from langgraph.prebuilt import create_react_agent
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.instrumentation.langchain_v2 import LangChainInstrumentor
from opentelemetry import trace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import json

from langchain_core.messages import ToolMessage

resource = Resource(attributes={"service.name": "langgraph-demo-simple"})

provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("langchain.app.tracer")
trace_propagator = TraceContextTextMapPropagator()
baggage_propagator = W3CBaggagePropagator()

LangChainInstrumentor().instrument(tracer_provider=provider)




class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
            except Exception as e:
                # Handle rate limit or other errors gracefully
                tool_result = {"error": f"Tool execution failed: {str(e)}. You may be rate limited."}

            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}




graph_builder = StateGraph(State)

tool = DuckDuckGoSearchResults()
tools = [tool]


llm = init_chat_model(
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",
)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END



tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

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


llm_with_tools = llm.bind_tools(tools)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
# graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
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
        # user_input = "What do you know about LangGraph?"
        # print("User: " + user_input)
        # stream_graph_updates(user_input)
        break
