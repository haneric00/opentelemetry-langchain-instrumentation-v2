"""Test OpenTelemetry instrumentation with LangChain agents using AWS Bedrock."""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor
# from opentelemetry.instrumentation.langchain_v2 import LangChainInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
import re
import os


def setup_opentelemetry():
    """Set up OpenTelemetry instrumentation."""
    print("Setting up OpenTelemetry instrumentation...")
    provider = TracerProvider()
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    # LangchainInstrumentor().instrument(tracer_provider=provider)


def simple_calculator(query: str) -> str:
    """A simple calculator tool for testing."""
    try:
        return str(eval(query))
    except:
        return "Invalid calculation"


def simple_search(query: str) -> str:
    """A simple search tool for testing."""
    return f"Search results for: {query}"


def test_agent_with_bedrock():
    """Test agent callbacks with AWS Bedrock."""
    print("\n" + "=" * 60)
    print("🧪 Testing Agent Callbacks with AWS Bedrock")
    print("=" * 60)

    # Create tools
    tools = [
        Tool(
            name="Calculator",
            func=simple_calculator,
            description="Useful for when you need to answer questions about math.",
        ),
        Tool(
            name="Search",
            func=simple_search,
            description="Useful for when you need to search for information.",
        ),
    ]

    # Create the prompt template for ReAct agent
    prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

    # Create Bedrock LLM
    # AWS credentials are automatically loaded from AWS CLI config (~/.aws/credentials)
    try:
        llm = ChatBedrock(
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            model_kwargs={
                "temperature": 0,
                "max_tokens": 1000,
            },
        )
    except Exception as e:
        print(f"❌ Error initializing Bedrock: {e}")
        print("Make sure you have:")
        print("  1. AWS CLI configured: run 'aws configure' to set up credentials")
        print("  2. Appropriate IAM permissions for Amazon Bedrock")
        print("  3. Access to the Claude model in your configured region")
        print("  4. The model is available in your region")
        return

    # Create the agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )

    # Run the agent
    try:
        result = agent_executor.invoke({"input": "What is 5 + 3?"})
        print(f"\n✅ Result: {result}")
        print(
            "\n🎯 The agent callbacks (on_agent_action and on_agent_finish) should have been triggered!"
        )
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run the agent callback test."""
    setup_opentelemetry()

    print("\n🚀 Testing OpenTelemetry instrumentation with LangChain agents")
    print(
        "Watch for 'Hook on_agent_action()' and 'Hook on_agent_finish()' in the output!\n"
    )

    test_agent_with_bedrock()

    print("\n" + "=" * 60)
    print("🏁 Test completed!")
    print("Check the console output above for agent callback hooks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
