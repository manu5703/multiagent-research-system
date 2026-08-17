from config import get_llm


print("Creating Claude...")

llm = get_llm()

print("Sending request...")

response = llm.invoke(
    "Say hello and explain agentic AI in one sentence."
)

print("\nCLAUDE RESPONSE:")
print(response.content)