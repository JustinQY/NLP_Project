from datasets import Dataset
from transformers import AutoTokenizer

def tokenize_scripts(scripts):
    print("Select GPT2-7b Version for Tokenization...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = chunk_scripts(tokenizer, scripts)
    return tokenizer, dataset


def chunk_scripts(tokenizer, scripts, model_name="gpt2-xl", max_len=1024, stride=512):
    print("Scripts Received! \nBegin to chunk scripts into pieces length less than 1024...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = []
    attention_masks = []
    labels = []

    for script in scripts:
        tokens = tokenizer(script, return_tensors="pt", truncation=False, padding=True)["input_ids"][0]
        total_len = tokens.size(0)
        
        for i in range(0, total_len - max_len + 1, stride):
            chunk = tokens[i:i + max_len]   
            input_ids.append(chunk.tolist())
            attention_masks.append([1] * len(chunk))
            labels.append(chunk.tolist())

    print("Chunking Finished! Ready to return the new scripts dataset...")
    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    })