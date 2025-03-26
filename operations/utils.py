import re
import random
from huggingface_hub import login
from google.colab import userdata
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig

#
# Hugging Face Login
#
def huggingface_login():
    hf_token = userdata.get('HF_TOKEN')

    if hf_token:
        login(token=hf_token)
        print("Hugging Face Successfully Login!")
    else:
        print("HF_TOKEN not found in environment variables.")

#
# Scripts Cleaner
#
def clean_script(script):
    lines = script.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("Written by") or line.startswith("[Scene"):
            continue
        if ':' in line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

#
# Scripts Tokenizer
#
def tokenize_scripts(scripts):
    print("Select GPT2-7b Version for Tokenization...\n")
    tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    tokens, dataset = chunk_scripts(tokenizer, scripts)
    return tokenizer, tokens, dataset

#
# Script Chunker By 1024 Length
#
def chunk_scripts(tokenizer, scripts, model_name="gpt2-xl", max_len=1024, stride=512):
    print("\nScripts Received! \n\nBegin to chunk scripts into pieces length less than 1024...\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    input_ids = []
    attention_masks = []
    labels = []
    tokens_list = []

    for script in scripts:
        tokens = tokenizer(script, return_tensors="pt", truncation=False, padding=True)["input_ids"][0]
        tokens_list.append(tokens)
        total_len = tokens.size(0)
        
        for i in range(0, total_len - max_len + 1, stride):
            chunk = tokens[i:i + max_len]   
            input_ids.append(chunk.tolist())
            attention_masks.append([1] * len(chunk))
            labels.append(chunk.tolist())

    print("\nChunking Finished! Ready to return the new scripts dataset...\n")
    return tokens_list, Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    })

def custom_trainer(model, dataset, tokenizer, lr=2e-4, warmup=0.03, L2=0.05, batch=1, epochs=3):
    print("Hyperparams Received! Started to generate Trainer...\n\n")
    training_arguments = TrainingArguments(
        report_to="none",
        output_dir="./output",
        # evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_ratio=0.03,
        weight_decay=0.05,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        # load_best_model_at_end=True,
        logging_steps=20,
        logging_strategy="steps",
        fp16=True,
        # optim="paged_adamw_8bit",
        # save_total_limit=3,
    )
    
    print(f"Training Arguments Generated: \n")
    # print(TrainingArguments)
    print("\nStarted to generate Trainer...\n")

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    print(f"Trainer Generated: \n")
    # print(trainer)
    return training_arguments, trainer


def generate_prompt(characters, location="Central Perk", scenario="having coffee", seed_dialogue=None, lines=10):
    """
    script-type prompt
    - characters: List[str]
    - location: str
    - seed_dialogue: Dict[str, str], beginning sentences (optional)
    - lines: int, num of lines of conversation
    """
    assert len(characters) >= 2, "Please assign at least 2 charactors~"

    prompt = f"[Scene: {location}, {', '.join(characters)} are {scenario}.]\n\n"

    line_count = 0

    if seed_dialogue:
        for speaker, line in seed_dialogue.items():
            prompt += f"{speaker}: {line.strip()}\n"
            line_count += 1

    speaker_cycle = characters * ((lines // len(characters)) + 2)
    idx = 0

    while line_count < lines:
        speaker = speaker_cycle[idx % len(speaker_cycle)]
        idx += 1
        # skip the given lines
        if seed_dialogue and speaker in seed_dialogue and line_count < len(seed_dialogue):
            continue
        prompt += f"{speaker}:\n"
        line_count += 1

    return prompt.strip()


def generate_script(prompt, max_new_tokens=500, temperature=0.9, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated
