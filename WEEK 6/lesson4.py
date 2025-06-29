from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

prompt = "The future of AI in healthcare"
suggestions = generator(prompt, max_length=50, num_return_sequences=3)
for i, suggestion in enumerate(suggestions):
    print(f"Suggestion {i+1}: {suggestion['generated_text']}")