# env OTEL_PYTHON_DISTRO=aws_distro \
#            OTEL_PYTHON_CONFIGURATOR=aws_configurator \
#            OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
#            OTEL_EXPORTER_OTLP_LOGS_HEADERS="x-aws-log-group=test/genesis,x-aws-log-stream=default,x-aws-metric-namespace=genesis" \
#            OTEL_RESOURCE_ATTRIBUTES="service.name=langchain-instrumentor-bedrock-test-app" \
#            AGENT_OBSERVABILITY_ENABLED="true" \
#            opentelemetry-instrument python  "/Users/ehrican/Desktop/langgraph sample/agent_v5.py"


import math
import types
import uuid
import os
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import (
    convert_positional_only_function_to_tool
)

import boto3
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from langgraph.prebuilt import create_react_agent
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.instrumentation.langchain_v2 import LangChainInstrumentor
from opentelemetry import trace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


from opentelemetry.instrumentation.langchain_v2.aws_auth_session import AwsAuthSession

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "agent_v5"

# Also set LangSmith variables for newer versions
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "agent_v5"


resource = Resource(attributes={"service.name": "langgraph-demo"})

provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("langchain.app.tracer")
trace_propagator = TraceContextTextMapPropagator()
baggage_propagator = W3CBaggagePropagator()


#########################################################################

# provider = TracerProvider()
# exporter = OTLPSpanExporter(
#     endpoint="https://xray.us-west-2.amazonaws.com/v1/traces",
#     session=AwsAuthSession("us-west-2")
# )
# processor = BatchSpanProcessor(exporter)
# provider.add_span_processor(processor)
# trace.set_tracer_provider(provider)
# trace_propagator = TraceContextTextMapPropagator()
# baggage_propagator = W3CBaggagePropagator()


LangChainInstrumentor().instrument()



# Collect functions from `math` built-in
all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(
        function, types.BuiltinFunctionType
    ):
        continue
    # This is an idiosyncrasy of the `math` library
    if tool := convert_positional_only_function_to_tool(
        function
    ):
        all_tools.append(tool)

# Create registry of tools. This is a dict mapping
# identifiers to tool instances.
tool_registry = {
    str(uuid.uuid4()): tool
    for tool in all_tools
}

# Index tool names and descriptions in the LangGraph
# Store. Here we use a simple in-memory store.
embeddings = init_embeddings("bedrock:cohere.embed-english-v3")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["description"],
    }
)
for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),
        tool_id,
        {
            "description": f"{tool.name}: {tool.description}",
        },
    )

bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2',  # Replace with your region
    )


# llm = init_chat_model("bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0")
llm = init_chat_model("bedrock:anthropic.claude-3-haiku-20240307-v1:0", temperature=0)

# llm = ChatBedrock(
#             model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
#             client=bedrock_client,
#         )

builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)
agent


query = "Use available tools to calculate arc cosine of 0.5."

# Test it out
for step in agent.stream(
    {"messages": query},
    stream_mode="updates",
):
    for _, update in step.items():
        for message in update.get("messages", []):
            message.pretty_print()
