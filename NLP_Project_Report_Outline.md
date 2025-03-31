
# 📝 NLP Final Project Report: TV Show Script Generation using LLaMA2 + LoRA

## 1. Introduction

### ✅ What I did:
- Fine-tuned a pre-trained language model (LLaMA-2-7b-hf) on the full script dataset of *Friends* TV Show (200 episodes).
- Generated new episode dialogues in the same style and structure as the original scripts.

### ✅ Why:
- To explore the capability of LLMs in mimicking character-specific dialogue and scene-driven scripts.
- To experiment with lightweight fine-tuning techniques (LoRA + 8-bit quantization) under limited hardware resources.

---

## 2. Dataset Collection & Preprocessing

### ✅ What I did:
- Collected, cleaned, and structured the entire Friends script dataset (200 episodes).
- Converted full scripts into episode-wise text segments with consistent formatting.

### ✅ How:
- Removed metadata (e.g., writer/transcriber notes) using a custom `clean_script()` function.
- Retained scene descriptions like `[Scene: ...]` and structured character dialogues `Character: Line`.
- Saved each episode as a clean, prompt-ready sample.

### ✅ Why:
- Clean formatting improves the learning of dialogue structure and flow.
- Scene and character cues provide valuable context for style-specific generation.

---

## 3. Model Selection & Fine-tuning Strategy

### ✅ What I did:
- Used `meta-llama/Llama-2-7b-hf` model as the backbone.
- Applied LoRA and BitsAndBytesConfig to optimize for low-memory fine-tuning.

### ✅ How:
- Loaded model in 8-bit using `bitsandbytes`, reduced GPU memory usage.
- Applied LoRA with target modules `["q_proj", "v_proj"]`, rank=8, alpha=32.
- Constructed training samples with tokenizer chunking from cleaned scripts.

### ✅ Why:
- LLaMA-2 is powerful and open-access.
- LoRA and 8-bit quantization reduce memory load while maintaining performance.

---

## 4. Script Generation Pipeline

### ✅ What I did:
- Designed a flexible prompt generator to trigger script creation.
- Controlled role, scene, and number of lines in each generated sample.

### ✅ How:
- Seed prompt includes `[Scene: ...]` + 1–2 initial lines of dialogue.
- Generated samples using `generate()` with `max_new_tokens` and temperature sampling.
- Post-processed to preserve dialogue structure.

### ✅ Why:
- Prompt engineering is critical to maintain structure and character consistency.
- Flexibility in prompt design enables diverse generation scenarios.

---

## 5. Evaluation

### ✅ What I did:
- Evaluated model performance using Perplexity, BLEU, ROUGE, and human-style scoring.

### ✅ How:
- Used `Trainer.evaluate()` and `math.exp(eval_loss)` for Perplexity.
- Computed BLEU and ROUGE using Huggingface `datasets` metrics.
- Created `evaluate_bleu_rouge.py` for repeatable evaluation.
- Developed a local manual rating function (`local_score()`) for subjective quality scoring.

### ✅ Why:
- Perplexity shows language modeling quality.
- BLEU/ROUGE provide quantitative feedback on similarity.
- Human scoring captures fluency, humor, and character style.

---

## 6. Results & Analysis

### ✅ What I did:
- Compared generated dialogues under different prompt and sampling settings.
- Assessed quality via structure, tone, and coherence.

### ✅ How:
- Measured style similarity with original scripts.
- Verified that character tone matches (e.g., Chandler’s sarcasm).
- Tested generation length, diversity, and repetition control.

### ✅ Why:
- Structural and stylistic alignment is key to believable TV scripts.
- Prompt setup and temperature impact the creativity and control of generated lines.

---

## 7. Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| Out of memory errors | Used LoRA + 8-bit quantization |
| Input length limit exceeded | Chunked long episodes into shorter training units |
| Output repetition | Optimized prompts and postprocessing |
| GPT-4 evaluation blocked | Used GPT-3.5 or local scoring instead |

---

## 8. Future Work

- ✅ Experiment with LLaMA-2-13B or Mixtral for higher-quality generation.
- ✅ Add more diverse script datasets (The Office, The Boys).
- ✅ Use GPT-4 or crowdsource for richer evaluations.
- ✅ Train models on full scenes with action + dialogue.
- ✅ Fine-tune with multi-turn dialogue for better context continuity.

---

## 9. Conclusion

- Successfully fine-tuned a powerful LLM to generate new TV script dialogues.
- Built a robust pipeline for cleaning, training, generating, and evaluating.
- Demonstrated style-aware, character-consistent generation results.
