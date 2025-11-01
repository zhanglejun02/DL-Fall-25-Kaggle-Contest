
import os
from unsloth import FastLanguageModel
import torch
import random, numpy as np, gc, hashlib, subprocess
from typing import List
from datasets import load_dataset

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


MAX_LEN_TRAIN = 1024  # Increase length to preserve more context
MAX_LEN_GEN = 64      # Generation only needs few tokens

def load_unsloth_local_model(local_model_path, max_seq_length=2048):
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = local_model_path,
        max_seq_length = max_seq_length,
        dtype = torch.float16,
        load_in_4bit = True,
        local_files_only = True,
        device_map = "auto",
        trust_remote_code = True,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Change to right padding!
    tokenizer.truncation_side = "left"

    print(f" Model loaded successfully! Padding side: {tokenizer.padding_side}")
    return model, tokenizer

LOCAL_MODEL_PATH = "Meta-Llama-3.1-8B"
model, tokenizer = load_unsloth_local_model(LOCAL_MODEL_PATH)

# -------------------------
# Key Fix 2: Simplified LoRA configuration
# -------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # Keep moderate
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,  # Lower alpha to avoid overfitting
    lora_dropout = 0.05,  # Lower dropout
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)
model.print_trainable_parameters()

def download_dataset_with_cli(dataset_name, local_dir=None):
    if local_dir is None:
        local_dir = f"./datasets/{dataset_name.replace('/', '_')}"
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    cmd = [
        "huggingface-cli", "download", dataset_name,
        "--repo-type", "dataset",
        "--local-dir", local_dir,
        "--local-dir-use-symlinks", "False",
    ]
    try:
        print(f"Start downloading dataset {dataset_name} to {local_dir}...")
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print("Download completed!")
        return local_dir
    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e}")
        return None

dataset_name = "ad6398/nyu-dl-teach-maths-comp"
local_path = download_dataset_with_cli(dataset_name)
full = load_dataset(local_path, split="train")
print(f"Dataset loaded successfully! Sample count: {len(full)}")

def qhash(q):
    return int(hashlib.md5(q.encode("utf-8")).hexdigest(), 16) % (10**9)

groups = [qhash(q) for q in full["question"]]
rng = np.random.default_rng(SEED)
unique_groups = np.array(sorted(set(groups)))
rng.shuffle(unique_groups)
cut = int(len(unique_groups) * 0.9)
train_group_set = set(unique_groups[:cut])

train_idx = [i for i, g in enumerate(groups) if g in train_group_set]
valid_idx = [i for i, g in enumerate(groups) if g not in train_group_set]

train_ds = full.select(train_idx)
valid_ds = full.select(valid_idx)
test_ds = load_dataset(local_path, split="test")
print(f"Training set: {len(train_ds)}, Validation set: {len(valid_ds)}, Test set: {len(test_ds)}")


def create_prompt_template(question, solution, label=None):
    """Create clearer prompt format"""
    prompt = f"""Please judge whether the following math solution is correct:

Question: {question}

Solution: {solution}

Please answer only "correct" or "incorrect":"""
    
    if label is not None:
        answer = "correct" if label == 1 else "incorrect"
        prompt += f"{answer}</s>"
    else:
        prompt += ""  # Leave empty for model to generate during inference
    
    return prompt

def fmt_train(b):
    texts = []
    for q, s, y in zip(b["question"], b["solution"], b["is_correct"]):
        texts.append(create_prompt_template(q, s, y))
    return {"text": texts}

def fmt_infer(b):
    texts = []
    for q, s in zip(b["question"], b["solution"]):
        texts.append(create_prompt_template(q, s))
    return {"text": texts}

train_formatted = train_ds.map(fmt_train, batched=True, remove_columns=train_ds.column_names)
valid_formatted = valid_ds.map(fmt_infer, batched=True, remove_columns=valid_ds.column_names)
test_formatted = test_ds.map(fmt_infer, batched=True, remove_columns=test_ds.column_names)

print("Training sample example:")
print(train_formatted[0]["text"][:500] + "...")


from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal language modeling
    pad_to_multiple_of=8
)


training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Use epochs instead of steps
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,  # Slightly higher learning rate
    warmup_ratio=0.1,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_formatted,
    eval_dataset=valid_formatted,
    dataset_text_field="text",
    max_seq_length=MAX_LEN_TRAIN,
    data_collator=data_collator,
    args=training_args,
    packing=False,  # No packing for more stable training
)


print("=== Verify data format ===")
sample_text = train_formatted[0]["text"]
sample_encoded = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=MAX_LEN_TRAIN)
print(f"Sample length: {len(sample_encoded['input_ids'][0])}")
print(f"Sample content preview: {sample_text[:200]}...")


print("===== Start training =====")
trainer.train()
print("===== Training completed =====")


def parse_model_output(text):
    """More robust output parsing"""
    text = text.lower().strip()
    
    # Search for keywords
    if "correct" in text or "true" in text or "1" in text:
        return 1
    elif "incorrect" in text or "false" in text or "0" in text:
        return 0
    
    # Default if no keywords found
    return 0

def predict(model, tokenizer, texts, batch_size=8):
    """Batch prediction"""
    model.eval()
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_LEN_TRAIN
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Only need to generate few tokens
                do_sample=False,    # Greedy decoding
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and parse
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for original, generated in zip(batch_texts, generated_texts):
            # Extract the model generated part
            generated_part = generated[len(original):]
            pred = parse_model_output(generated_part)
            predictions.append(pred)
    
    return predictions


print("===== Saving model =====")
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")


print("===== Start validation =====")
valid_texts = valid_formatted["text"]
valid_true = valid_ds["is_correct"]

valid_preds = predict(model, tokenizer, valid_texts)

from sklearn.metrics import accuracy_score, f1_score, classification_report
accuracy = accuracy_score(valid_true, valid_preds)
f1 = f1_score(valid_true, valid_preds)

print(f"Validation accuracy: {accuracy:.4f}")
print(f"Validation F1 score: {f1:.4f}")
print("\nClassification report:")
print(classification_report(valid_true, valid_preds))

# -------------------------
# Test set prediction examples
# -------------------------
print("===== Test set prediction examples =====")
test_texts = test_formatted["text"][:5]  # First 5 samples
test_predictions = predict(model, tokenizer, test_texts)

for i, (text, pred) in enumerate(zip(test_texts, test_predictions)):
    print(f"Sample {i+1}:")
    print(f"Prediction: {pred} ({'correct' if pred == 1 else 'incorrect'})")
    print(f"Question preview: {text[:100]}...")
    print("-" * 50)

print("Training and evaluation completed!")
