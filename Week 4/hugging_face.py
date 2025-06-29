from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


inputs = tokenizer("summarize: " + code, return_tensors="pt")

summary_ids = model.generate(**inputs, max_length=50)

print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
