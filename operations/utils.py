import re
import math
import torch
import random

from datasets import Dataset
from huggingface_hub import login
from google.colab import userdata
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, EarlyStoppingCallback

gpt2 = "gpt2-xl"
llama_2_7 = "meta-llama/Llama-2-7b-hf"
llama_2_13 = "meta-llama/Llama-2-13b-hf"

mixtral_87 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
nousH_13 = "NousResearch/Nous-Hermes-13b"

output_path = "/content/drive/MyDrive/NLP_Project/model"

# Hugging Face Login
def huggingface_login():
    hf_token = userdata.get('HF_TOKEN')

    if hf_token:
        login(token=hf_token)
        print("Hugging Face Successfully Login!")
    else:
        print("HF_TOKEN not found in environment variables.")


#########################
#   Data Preprocessing  #
#########################

class DataPreprocessor:
    def __init__(self, tokenizer_name):
        print("Initialize data propressor...")
        self.train_set = None
        self.val_set = None
        self.tokenizer_name = tokenizer_name
        self.tokenizer = self.create_tokenizer()


    def create_tokenizer(self):
        print(f"Select {self.tokenizer_name} for Tokenization...\n")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def clean_script(self, script):
        lines = script.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Written by") or line.startswith("Transcribed by"):
                continue
            if line.startswith("[") or line.startswith("("):
                cleaned_lines.append(line)
            elif ':' in line:
                cleaned_lines.append(line)
            elif line.isupper() and len(line.split()) > 2:
                cleaned_lines.insert(0, line)
        
        cleaned_lines.append("[End of episode]")
        eos_token = self.tokenizer.eos_token or "<eos>"
        cleaned_lines.append(eos_token)
        
        return "\n".join(cleaned_lines)
    

    def data_split(self, df):
        train_scripts_df, val_scripts_df = train_test_split(df, test_size=0.1, random_state=42)
        train_scripts = train_scripts_df.tolist()
        val_scripts = val_scripts_df.tolist()
        return train_scripts, val_scripts
    

    def tokenize_scripts(self, train_scripts, val_scripts):
        print("Tokenizing [training] scripts...\n")
        train_tokens, train_dataset = self.chunk_scripts(train_scripts)
        print("Tokenizing [validation] scripts...\n")
        val_tokens, val_dataset = self.chunk_scripts(val_scripts)
        print("\nTokenizations all done!")
        self.train_set = train_dataset
        self.val_set = val_dataset
        return train_tokens, val_tokens, train_dataset, val_dataset


    # Script Chunker By 1024 Length
    def chunk_scripts(self, scripts, max_len=1024, stride=512):
        print("\nScripts Received! \n\nBegin to chunk scripts into pieces length less than 1024...\n")

        input_ids = []
        attention_masks = []
        labels = []
        tokens_list = []

        for script in scripts:
            tokens = self.tokenizer(script, return_tensors="pt", truncation=False, padding=True)["input_ids"][0]
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


##############################
#  Custom Pre-trained Model  #
##############################

class CustomModel:
    def __init__(self, model_name, tokenizer, train_dataset, val_dataset, lr, warmup, L2, batch_size, epochs, enable_lora, enable_bitsbytes):
        print("Initialize custom pretrained model...")
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.lora = None
        self.bitsbytes = None
        self.model = self.custom_pretrained_model(enable_lora, enable_bitsbytes)
        self.training_args, self.trainer = self.custom_trainer(train_dataset, val_dataset, lr, warmup, L2, batch_size, epochs)

    # Customize Pre-trained Model
    def custom_pretrained_model(self, enable_lora, enable_bitsbytes):
        if enable_bitsbytes==True:
            self.bitsbytes = self.bits_bytes_config()
            model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=self.bitsbytes)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if enable_lora == True:
            self.lora = self.lora_config()
            model = get_peft_model(model, self.lora)
        
        return model

    # LoRA Config
    def lora_config(self):
        if self.model_name == gpt2:
            target_modules = ["c_attn"]
        elif self.model_name == llama_2_7:
            target_modules = ["q_proj", "v_proj"]
        elif self.model_name == llama_2_13:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif self.model_name == mixtral_87:
            target_modules = ["Wq", "Wk", "Wv", "Wo", "W1", "W2"]
        elif self.model_name == nousH_13:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias='none',
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM
        )
        return config

    # Bits Bytes Config
    def bits_bytes_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        return bnb_config

    # Create arguments and trainer for model training
    def custom_trainer(self, train_dataset, val_dataset, lr, warmup, L2, batch_size, epochs):
        print("Hyperparams Received! Started to generate Trainer...\n\n")
        training_arguments = TrainingArguments(
            learning_rate=lr,
            lr_scheduler_type="constant",
            warmup_ratio=warmup,
            weight_decay=L2,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            label_names = ["labels"],

            eval_steps=250,
            eval_strategy="steps",
            per_device_eval_batch_size=batch_size,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            save_steps=500,
            save_strategy="steps",

            logging_steps=250,
            logging_strategy="steps",
            fp16=True,
            # optim="paged_adamw_8bit",

            report_to="none",
            output_dir=output_path,
        )

        print("\nStarted to generate Trainer...\n")

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )

        print(f"[Training Arguments] and [Trainer] Generated! \n")
        return training_arguments, trainer


########################
#   Model Generation   #
########################

class ScriptGenerator:
    def __init__(self):
        print("Initialize TV Show Scripts generator...")
        self.prompt = None
        self.generated_scripts = []


    def create_prompt(self, characters, location="Central Perk", scenario="having coffee", seed_dialogue=None, continue_speaker=None):
        assert len(characters) >= 2, "Please assign at least 2 charactors~"

        prompt = f"""
        You are going to generate a new episode of the show *Friends*.
        
        The episode should include multiple scenes, natural conversations, character-specific humor, and a clear ending.
        
        [Scene: {location}, {', '.join(characters)} are {scenario}.]\n\n
        """

        if seed_dialogue:
            for speaker, line in seed_dialogue.items():
                prompt += f"{speaker}: {line.strip()}\n"

        if continue_speaker:
            prompt += f"{continue_speaker}:"

        self.prompt = prompt.strip()
        return self.prompt


    # New Script Generator
    def create_new_script(self, model, tokenizer, max_new_tokens=2048, temperature=0.9, top_k=50, top_p=0.95):
        model.eval()
        inputs = tokenizer(self.prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens = max_new_tokens,
                do_sample = True,
                temperature = temperature,
                top_k = top_k,
                top_p = top_p,
                repetition_penalty = 1.2,
                pad_token_id = tokenizer.eos_token_id,
                eos_token_id = tokenizer.eos_token_id,
            )

        new_script = tokenizer.decode(output[0], skip_special_tokens=True)
        self.generated_scripts.append(new_script)
        return new_script


    def pretty_print_script(self, script_str):
        lines = script_str.strip().split("\n")

        for line in lines:
            line = line.strip()

            if line.startswith("[Scene") or line.startswith("["):
                print(line + "\n" + "-" * len(line))
            elif ":" in line:
                speaker, dialogue = line.split(":", 1)
                print(f"{speaker.strip()}: {dialogue.strip()}")
            elif line.startswith("(") and line.endswith(")"):
                print(f"   {line}")
            else:
                print(line)


########################
#   Model Evaluation   #
########################

class CustomEvaluator:
    def __init__(self, trainer):
        self.trainer = trainer


    def perplexity(self):
        evaluate_results = self.trainer.evaluate()
        loss = evaluate_results["eval_loss"]
        perplexity = math.exp(loss)
        return perplexity


####################
#   Model Loading  #
####################

def load_custom_model(enable_lora, path):
    if enable_lora==True:
        config = PeftConfig.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        lora_model = PeftModel.from_pretrained(model, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(path)
    
    return model
