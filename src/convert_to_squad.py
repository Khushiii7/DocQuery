import csv
import json
import sys

INPUT_CSV = 'data/qa_annotation_template.csv'
OUTPUT_JSON = 'data/hf_doc_qa_manual_squad.json'

def main():
    data = []
    with open(INPUT_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            context = row['context']
            question = row['question']
            answer = row['answer']
            answer_start = context.find(answer)
            if answer_start == -1:
                print(f"Warning: answer not found in context for question: {question}")
                print(f"Context repr: {repr(context)}")
                print(f"Answer repr: {repr(answer)}")
            qa = {
                "context": context,
                "question": question,
                "answers": [{
                    "text": answer,
                    "answer_start": answer_start
                }]
            }
            data.append(qa)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as out:
        json.dump({"data": data}, out, indent=2, ensure_ascii=False)
    print(f"Wrote {len(data)} QA pairs to {OUTPUT_JSON}")

if __name__ == '__main__':
    main() 