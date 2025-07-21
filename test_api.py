import requests

API_URL = "http://127.0.0.1:8000/query"

test_cases = [
    {
        "question": "How do I install Transformers using pip?",
        "context": "To install Transformers with pip, run: pip install transformers. This will install the package in your Python environment."
    },
    {
        "question": "Which machine learning frameworks are supported by Transformers?",
        "context": "Transformers works with PyTorch, TensorFlow 2.0, and Flax. It has been tested on Python 3.9+, PyTorch 2.1+, TensorFlow 2.6+, and Flax 0.4.1+."
    },
    {
        "question": "Why should I use a virtual environment when working with Transformers?",
        "context": "A virtual environment helps manage different projects and avoids compatibility issues between dependencies. Take a look at the Install packages in a virtual environment using pip and venv guide if you're unfamiliar with Python virtual environments."
    },
    {
        "question": "What is the purpose of the Trainer class in Transformers?",
        "context": "The Trainer class provides an API for feature-complete training in PyTorch, and it supports distributed training on multiple GPUs/TPUs, mixed precision for NVIDIA GPUs, AMD GPUs, and torch.amp for PyTorch. Trainer goes hand-in-hand with the TrainingArguments class, which offers a wide range of options to customize how a model is trained."
    },
    {
        "question": "What is the purpose of pipelines in Transformers?",
        "context": "The pipelines are a great and easy way to use models for inference. These pipelines are objects that abstract most of the complex code from the library, offering a simple API dedicated to several tasks, including Named Entity Recognition, Masked Language Modeling, Sentiment Analysis, Feature Extraction and Question Answering."
    },
]

for i, payload in enumerate(test_cases, 1):
    response = requests.post(API_URL, json=payload)
    print(f"Test case {i}:")
    print("Question:", payload["question"])
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
    print("-" * 40) 