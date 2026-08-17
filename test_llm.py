from config import get_llm


llm = get_llm()

response = llm.invoke(
    "What is agentic AI? Answer in one sentence."
)

print("\nCLAUDE RESPONSE:")
print(response.content)