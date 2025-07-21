import json
import os

def create_squad_dataset():
    processed_dir = "C:/Users/khush/Documents/Project/DocQuery/data/processed"
    dataset = {"data": []}

    for filename in os.listdir(processed_dir):
        with open(os.path.join(processed_dir, filename), "r", encoding="utf-8") as f:
            context = f.read()

            qa = {
                "context": context,
                "question": "How do I initialize the BERT tokenizer?",
                "answers": [{
                    "text": "BertTokenizer.from_pretrained('bert-base-uncased')",
                    "answer_start": context.find("BertTokenizer.from_pretrained('bert-base-uncased')")
                }]
            }
            dataset["data"].append(qa)

    with open("C:/Users/khush/Documents/Project/DocQuery/data/hf_doc_qa_dataset.json", "w", encoding="utf-8") as out:
        json.dump(dataset, out, indent=4)

if __name__ == "__main__":
    create_squad_dataset()
