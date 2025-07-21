from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# === Fill these in with your test case ===
question = "What is the purpose of pipelines in Transformers?"
context = """
The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering.
"""

# Use the same model as your API
model_name = "deepset/roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    max_length=512,
    truncation="only_second"
)

with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
answer_start = torch.argmax(start_logits)
answer_end = torch.argmax(end_logits) + 1

answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
)

print(f"Question: {question}")
print(f"Answer: {answer.strip()}")
print(f"Context: {context.strip()}") 