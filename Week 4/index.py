import openai

openai_api_key = "gffgfg a"
client = openai.OpenAI(api_key=openai_api_key)

def generate_tests(code_snippet: str) -> str:
    """
    Send the code to Codex and ask for pytest tests.
    """
    prompt = f"Write pytest tests for this code:\n\n{code_snippet}\n\n# tests\n"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",            
        prompt=prompt,
        max_tokens=200,
        temperature=0.2,
        n=1,
        stop=["# end"]
    )
    return response.choices[0].text.strip()

sample = """
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""

print(generate_tests(sample))


