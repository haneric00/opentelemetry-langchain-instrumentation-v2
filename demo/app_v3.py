import boto3
from langchain_aws import ChatBedrockConverse, ChatBedrock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks.manager import CallbackManager

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.instrumentation.langchain_v2 import LangChainInstrumentor
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

# Create a serializable wrapper for the bedrock client
class SerializableBedrock:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        return getattr(self._client, name)

    def __getstate__(self):
        return {"region_name": "us-east-1", "service_name": "bedrock-runtime"}

    def __setstate__(self, state):
        pass

resource = Resource(attributes={"service.name": "langchain-demo"})
provider = TracerProvider(resource=resource)
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

LangChainInstrumentor().instrument()
# LangchainInstrumentor().instrument()


# Initialize AWS Bedrock client with the serializable wrapper
bedrock_client = SerializableBedrock(boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
))

# Simple calculator function
def calculator(expression):
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

# Main assistant class - refactored to match how your BedrockLLM version works
class BedrockAssistant:
    def __init__(self):
        self.llm = ChatBedrock(
            client=bedrock_client,
            model_id="anthropic.claude-v2",
            temperature=0.7,
            max_tokens=1000
        )
        self.messages = [
            SystemMessage(content="""You are a helpful AI assistant with calculation abilities.
            When you need to calculate something, format your response like this:
            CALCULATE: [mathematical expression]
            Wait for the calculation result before continuing.""")
        ]
        self.metadata = {"agent_name": "BedrockChatAssistant"}

    def invoke(self, inputs, metadata=None):
        # Add user message
        user_input = inputs["input"]
        self.messages.append(HumanMessage(content=user_input))

        # Use the metadata pattern that works in your BedrockLLM example
        combined_metadata = {**(metadata or {}), **self.metadata}

        # Get response from LLM
        response = self.llm.invoke(
            self.messages,
            config={"callbacks": CallbackManager(handlers=[], metadata=combined_metadata)}
        )
        response_content = response.content

        # Process calculation if needed
        if "CALCULATE:" in response_content:
            expression = response_content.split("CALCULATE:")[1].split("\n")[0].strip()
            calc_result = calculator(expression)

            self.messages.append(AIMessage(content=response_content))
            self.messages.append(HumanMessage(content=f"Calculation result: {calc_result}"))

            final_response = self.llm.invoke(
                self.messages,
                config={"callbacks": CallbackManager(handlers=[], metadata=combined_metadata)}
            )
            final_content = final_response.content

            self.messages.append(AIMessage(content=final_content))
            return {"output": f"[Used calculator] {final_content}"}
        else:
            self.messages.append(AIMessage(content=response_content))
            return {"output": response_content}

# Main function following your BedrockLLM pattern
def main():
    print("Welcome to the Bedrock Chat Assistant!")
    print("This assistant can chat and calculate with proper tracing.")
    print("Type 'exit' to quit.")
    print("-" * 50)

    assistant = BedrockAssistant()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break

        try:
            # Use the same invoke pattern as your BedrockLLM example
            response = assistant.invoke(
                {"input": user_input},
                {"metadata": {"interaction_id": "user_query"}}
            )
            print(f"\nAssistant: {response['output']}")
        except Exception as e:
            import traceback
            print(f"\nError: {str(e)}")
            traceback.print_exc()
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    main()
