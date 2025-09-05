# env OTEL_PYTHON_DISTRO=aws_distro \
#            OTEL_PYTHON_CONFIGURATOR=aws_configurator \
#            OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
#            OTEL_EXPORTER_OTLP_LOGS_HEADERS="x-aws-log-group=test/genesis,x-aws-log-stream=default,x-aws-metric-namespace=genesis" \
#            OTEL_RESOURCE_ATTRIBUTES="service.name=langchain-instrumentor-bedrock-test-app" \
#            AGENT_OBSERVABILITY_ENABLED="true" \
#            opentelemetry-instrument python /Users/ehrican/Desktop/opentelemetry-langchain-instrumentation-v2/demo/app.py

# env OTEL_SERVICE_NAME="langchain-instrumentor-bedrock-test-app" \
#     OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
#     AGENT_OBSERVABILITY_ENABLED="true" \
#     opentelemetry-instrument python /Users/ehrican/Desktop/opentelemetry-langchain-instrumentation-v2/demo/app.py

# env OTEL_SERVICE_NAME="langchain-instrumentor-bedrock-test-app" \
#     OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
#     AGENT_OBSERVABILITY_ENABLED="true" \
#     python /Users/ehrican/Desktop/opentelemetry-langchain-instrumentation-v2/demo/app.py

import asyncio
import os
from langchain_aws import BedrockLLM, ChatBedrock, ChatBedrockConverse
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from typing import TypedDict
from langgraph.graph import StateGraph
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults

import boto3

from opentelemetry import trace, baggage

from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from opentelemetry.instrumentation.langchain_v2 import LangChainInstrumentor


resource = Resource(attributes={"service.name": "langchain-demo"})


provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)


tracer = trace.get_tracer("langchain.app.tracer")
trace_propagator = TraceContextTextMapPropagator()
baggage_propagator = W3CBaggagePropagator()


####################################################################################################


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

####################################################################################################


LangChainInstrumentor().instrument()

bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2'  # Replace with your region
    )

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "agent_v6"

llm = BedrockLLM(
    client=bedrock_client,
    model_id="anthropic.claude-v2",
    # model_kwargs={
    #     "max_tokens_to_sample": 500,
    #     "temperature": 0.7,
    # },
    temperature=0.7,
    max_tokens=500
)

chatBed = ChatBedrock(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    # model_kwargs={
    #     "max_tokens_to_sample": 500,
    #     "temperature": 0.7,
    # },
    temperature=0.7,
    max_tokens=500
)


chatBedConv = ChatBedrockConverse(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    # model_kwargs={
    #     "max_tokens_to_sample": 500,
    #     "temperature": 0.7,
    # },
    temperature=0.7,
    max_tokens=500
)



def create_chatbot():
    prompt_template = """The following is a friendly conversation between a human and an AI assistant.

                    Current conversation:
                    {history}
                    Human: {input}
                    Assistant:"""

    PROMPT = PromptTemplate(
        input_variables=["history", "input"],
        template=prompt_template
    )

    memory = ConversationBufferMemory(human_prefix="Human", ai_prefix="Assistant")

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=PROMPT,
        verbose=False,
    )

    return conversation

def chat():
    chatbot = create_chatbot()

    print("Chatbot: Hello! How can I help you today? (Type 'quit' to exit)")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break

        response = chatbot.predict(input=user_input)
        print(f"Agent: {response}")


def create_chatBedConvAgent():
    search = DuckDuckGoSearchRun()

    def add(query):
        """Add two numbers together."""
        try:
            a, b = map(float, query.split(','))
            return f"{a} + {b} = {a + b}"
        except Exception as e:
            return f"Error with input '{query}': {str(e)}"

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for searching the internet for information."
        ),
        Tool(
            name="Addition",
            func=add,
            description="Adds two numbers together. Input should be two numbers separated by a comma."
        )
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=chatBedConv,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        memory=memory,
        handle_parsing_errors=False,
        max_iterations=3  # Reduce this to prevent potential loops
    )

    return agent


def create_agent():
    search = DuckDuckGoSearchRun()

    def add(query):
        """Add two numbers together."""
        try:
            a, b = map(float, query.split(','))
            return f"{a} + {b} = {a + b}"
        except Exception as e:
            return f"Error with input '{query}': {str(e)}"

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for searching the internet for information."
        ),
        Tool(
            name="Addition",
            func=add,
            description="Adds two numbers together. Input should be two numbers separated by a comma."
        )
    ]

    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=False,
        # memory=memory,
        handle_parsing_errors=False,
        max_iterations=3  # Reduce this to prevent potential loops
    )

    return agent


def agent():
    agentbot = create_agent()

    print("Agent: Hello! How can I help you today? (Type 'quit' to exit)")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Agent: Goodbye!")
            break

        try:
            response = agentbot.invoke(input=user_input)
            # print(f"Agent: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


async def main():
    print("Welcome to the AI Assistant!")
    print("1. Regular Chat (Direct LLM interaction)")
    print("2. Agent with Web Search")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        print("\nStarting regular chat mode...\n")
        chat()
    elif choice == '2':
        print("\nStarting agent with web search...\n")
        agent = create_agent()

        print("You can now chat with the agent. Type 'exit' to quit.")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break

            try:
                # response = agent.invoke({"input": user_input})
                response = agent.invoke(
                    {"input": user_input},
                    {"metadata": {"agent_name": "ConversationalAgent"}}
                )
                print(f"\nAgent: {response['output']}")
            except Exception as e:
                print(f"\nError: {e}")
                print("The agent encountered an error. Please try again.")



        bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-west-2'  # Replace with your region
        )


        # search = DuckDuckGoSearchResults()
        # tools = [search]
        # model = ChatBedrock(
        #     model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        #     region_name="us-west-2",
        #     temperature=0.9,
        #     max_tokens=2048,
        #     model_kwargs={
        #         "top_p": 0.9,
        #     },
        #     client=bedrock_client,
        # )

        # prompt = hub.pull(
        #     "hwchase17/openai-functions-agent",
        #     api_key=os.environ["LANGSMITH_API_KEY"],
        # )

        # agent = create_tool_calling_agent(model, tools, prompt)
        # agent_executor = AgentExecutor(agent=agent, tools=tools)

        # agent_executor.invoke({"input": "When was Amazon founded?"})



    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    # main()
    asyncio.run(main())
