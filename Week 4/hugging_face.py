from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load a pretrained code-summarization model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

code = """
def calculate_tax(income, threshold=35000):
    \"""
    Calculates tax owed: 0% up to threshold,
    20% above threshold.
    \"""
    if income <= threshold:
        return 0
    return (income - threshold) * 0.2
""" 


# 3. Tokenize: wrap code in a prompt
inputs = tokenizer("summarize: " + code, return_tensors="pt")

# 4. Generate summary tokens
summary_ids = model.generate(**inputs, max_length=50)

# 5. Decode to text
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
