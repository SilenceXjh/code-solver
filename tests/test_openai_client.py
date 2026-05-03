from code_solver.llm.openai_client import OpenAIClient

llm = OpenAIClient(model="deepseek-v4-flash", api_base="https://api.deepseek.com")

resp_content = llm.chat_simple(system="You are a helpful assistant", user="Who are you?")
print("response content:", resp_content)

usage = llm.get_usage()
print("usage:", usage)
