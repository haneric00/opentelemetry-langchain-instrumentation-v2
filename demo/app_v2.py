from langchain_aws import BedrockLLM, ChatBedrock
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import Tool
from langchain_core.tools import tool

from langchain.agents import initialize_agent, AgentType, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun  # Corrected import
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper  # Corrected import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

import boto3
import time
import datetime
import math
import requests
import json
import random

from opentelemetry import trace, baggage
from opentelemetry.context import Context, attach, detach
from opentelemetry.context import context as context_api
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.instrumentation.langchain_v2 import LangChainInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
# from openinference.instrumentation.langchain import LangChainInstrumentor

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain_v2.aws_auth_session import AwsAuthSession


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

resource = Resource(attributes={"service.name": "langchain-demo"})

provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("langchain.app.tracer")
trace_propagator = TraceContextTextMapPropagator()
baggage_propagator = W3CBaggagePropagator()

LangChainInstrumentor().instrument()
# LangchainInstrumentor().instrument()

bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2'  # Replace with your region
    )

llm = BedrockLLM(
    client=bedrock_client,
    model_id="anthropic.claude-v2",
    model_kwargs={
        "max_tokens_to_sample": 500,
        "temperature": 0.7,
    },
)

# llm = ChatBedrock(
#     client=bedrock_client,
#     model_id="bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0",
#         model_kwargs={
#             "max_tokens_to_sample": 500,
#             "temperature": 0.7,
#         },
# )

def create_chatbot():
    # Same implementation as before
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
        verbose=True,
    )

    return conversation

def chat():
    # Same implementation as before
    chatbot = create_chatbot()

    print("Chatbot: Hello! How can I help you today? (Type 'quit' to exit)")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break

        response = chatbot.predict(input=user_input)
        print(f"Agent: {response}")

def create_agent():
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    @tool
    def add(query: str) -> str:
        """Add two numbers together. The input should be two numbers separated by a comma."""
        try:
            a, b = map(float, query.split(','))
            return f"{a} + {b} = {a + b}"
        except Exception as e:
            return f"Error with input '{query}': {str(e)}"

    @tool
    def multiply(query: str) -> str:
        """Multiply two numbers. The input should be two numbers separated by a comma."""
        try:
            a, b = map(float, query.split(','))
            return f"{a} × {b} = {a * b}"
        except Exception as e:
            return f"Error with input '{query}': {str(e)}"

    @tool
    def get_current_time() -> str:
        """Get the current date and time."""
        now = datetime.datetime.now()
        return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}"

    @tool
    def calculate_square_root(number: str) -> str:
        """Calculate the square root of a number."""
        try:
            num = float(number.strip())
            if num < 0:
                return "Cannot calculate square root of a negative number in the real number system."
            result = math.sqrt(num)
            return f"The square root of {num} is {result}"
        except Exception as e:
            return f"Error calculating square root: {str(e)}"

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location. Use a city name."""

        # This is a mock implementation
        weather_options = ["sunny", "cloudy", "rainy", "snowy", "windy", "partly cloudy"]
        temp = round(random.uniform(0, 35), 1)
        condition = random.choice(weather_options)
        return f"Weather in {location}: {condition}, {temp}°C"

    tools = [
        search,
        wikipedia,
        add,
        multiply,
        get_current_time,
        calculate_square_root,
        get_weather
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return agent

def create_complex_agent():
    """Create a more complex agent with multiple tools in a sophisticated chain"""
    import random

    # Basic tools
    search = DuckDuckGoSearchRun()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # Math and utility tools
    @tool
    def add(query: str) -> str:
        """Add two numbers together. The input should be two numbers separated by a comma."""
        try:
            a, b = map(float, query.split(','))
            return f"{a} + {b} = {a + b}"
        except Exception as e:
            return f"Error with input '{query}': {str(e)}"

    @tool
    def multiply(query: str) -> str:
        """Multiply two numbers. The input should be two numbers separated by a comma."""
        try:
            a, b = map(float, query.split(','))
            return f"{a} × {b} = {a * b}"
        except Exception as e:
            return f"Error with input '{query}': {str(e)}"

    @tool
    def get_current_time(dummy: str = "") -> str:
        """Get the current date and time. You don't need to provide any input."""
        now = datetime.datetime.now()
        return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}"

    @tool
    def calculate_square_root(number: str) -> str:
        """Calculate the square root of a number."""
        try:
            num = float(number.strip())
            if num < 0:
                return "Cannot calculate square root of a negative number in the real number system."
            result = math.sqrt(num)
            return f"The square root of {num} is {result}"
        except Exception as e:
            return f"Error calculating square root: {str(e)}"

    @tool
    def get_weather(location: str) -> str:
        """Get the current weather for a location. Use a city name."""
        # This is a mock implementation
        weather_options = ["sunny", "cloudy", "rainy", "snowy", "windy", "partly cloudy"]
        temp = round(random.uniform(0, 35), 1)
        condition = random.choice(weather_options)
        return f"Weather in {location}: {condition}, {temp}°C"

    @tool
    def calculate_age(birth_year: str) -> str:
        """Calculate age from birth year."""
        try:
            birth_year = int(birth_year.strip())
            current_year = datetime.datetime.now().year
            age = current_year - birth_year
            return f"A person born in {birth_year} would be approximately {age} years old in {current_year}."
        except Exception as e:
            return f"Error calculating age: {str(e)}"

    tools = [
        search,
        wikipedia,
        add,
        multiply,
        get_current_time,
        calculate_square_root,
        get_weather,
        calculate_age
    ]

    # Create a specialized memory that also stores entities for the complex agent
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Initialize the complex agent
    complex_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=6,  # Allow more iterations for complex reasoning
    )

    return complex_agent


def main():
    """Main entry point that allows user to choose between chat, simple agent or complex agent."""
    print("Welcome to the AI Assistant!")
    print("----------------------------")
    print("1. Regular Chat (Direct LLM interaction)")
    print("2. Simple Agent with Web Search")
    print("3. Complex Agent with Multiple Tools")
    print("----------------------------")

    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            print("\nStarting regular chat mode...\n")
            chat()
            break
        elif choice == '2':
            print("\nStarting simple agent with web search...\n")
            agent = create_agent()

            print("You can now chat with the agent. Type 'exit' to quit.")

            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break

                try:
                    response = agent.invoke({"input": user_input})
                    print(f"\nAgent: {response['output']}")
                except Exception as e:
                    print(f"\nError: {e}")
                    print("The agent encountered an error. Please try again.")

            break
        elif choice == '3':
            print("\nStarting complex agent with multiple tools...\n")
            complex_agent = create_complex_agent()

            print("You can now chat with the complex agent. Type 'exit' to quit.")

            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Goodbye!")
                    break

                try:
                    response = complex_agent.invoke({"input": user_input})
                    print(f"\nComplex Agent: {response['output']}")
                except Exception as e:
                    print(f"\nError: {e}")
                    print("The complex agent encountered an error. Please try again.")

            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
