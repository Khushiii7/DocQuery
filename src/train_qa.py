from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

model_name = "bert-base-uncased"

dataset = load_dataset('json', data_files='data/hf_doc_qa_manual_squad.json', field='data')

# Simple train/validation split (80/20)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    questions = examples["question"]
    contexts = examples["context"]
    answers = examples["answers"]

    inputs = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(inputs["offset_mapping"]):
        answer_dict = answers[i][0] if isinstance(answers[i], list) else answers[i]
        answer = answer_dict["text"]
        answer_start = answer_dict["answer_start"]
        sequence_ids = inputs.sequence_ids(i)
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)
        start_char = answer_start
        end_char = answer_start + len(answer)
        token_start = token_end = None
        for idx in range(context_start, context_end):
            if offsets[idx][0] <= start_char < offsets[idx][1]:
                token_start = idx
            if offsets[idx][0] < end_char <= offsets[idx][1]:
                token_end = idx
        if token_start is None:
            token_start = context_start
        if token_end is None:
            token_end = context_end - 1
        start_positions.append(token_start)
        end_positions.append(token_end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)

model = AutoModelForQuestionAnswering.from_pretrained(model_name)

args = TrainingArguments(
    output_dir="models/bert-qa-hf",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("models/bert-qa-hf")
tokenizer.save_pretrained("models/bert-qa-hf")
