import os
from datasets import load_dataset, load_from_disk
from paths import DATASETS_DIR

# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def get_local_dataset_path(dataset_name: str, cache_dir: str = None) -> str:
    """
    Build a safe local path for storing datasets based on their Hugging Face name.

    Args:
        dataset_name (str): Hugging Face dataset identifier (e.g., 'knkarthick/samsum').
        cache_dir (str | None): Optional cache directory override (e.g., from config).

    Returns:
        str: Absolute path to local dataset folder.
    """
    safe_name = dataset_name.replace("/", "_").replace(":", "_")
    base_dir = cache_dir or DATASETS_DIR
    return os.path.join(base_dir, safe_name)


def select_subset(dataset, n_samples, seed=42):
    """
    Select a subset of the dataset.
    If n_samples is "all" or None, return the entire dataset.
    Otherwise, sample n_samples examples.
    """
    if n_samples == "all" or n_samples is None:
        return dataset
    
    if n_samples > len(dataset):
        print(f"⚠️  Requested {n_samples} samples but only {len(dataset)} available. Using all samples.")
        return dataset
    
    return dataset.shuffle(seed=seed).select(range(n_samples))

def load_and_prepare_dataset(cfg):
    """
    Load dataset splits according to configuration.
    Ensures the FULL dataset is cached, and subsets are selected per run.
    Supports both new-style ("dataset": {"splits": {...}}) and old-style (top-level keys) configs.
    """
    # -----------------------------------------------------------------------
    # Extract dataset configuration
    # -----------------------------------------------------------------------
    if "dataset" in cfg:
        cfg_dataset = cfg["dataset"]
        dataset_name = cfg_dataset["name"]
        splits_cfg = cfg_dataset.get("splits", {})
        n_train = splits_cfg.get("train", "all")
        n_val = splits_cfg.get("validation", "all")
        n_test = splits_cfg.get("test", "all")
        seed = cfg_dataset.get("seed", 42)
    elif "datasets" in cfg and isinstance(cfg["datasets"], list):
        cfg_dataset = cfg["datasets"][0]
        dataset_name = cfg_dataset["path"]
        n_train = cfg.get("train_samples", "all")
        n_val = cfg.get("val_samples", "all")
        n_test = cfg.get("test_samples", "all")
        seed = cfg.get("seed", 42)
    else:
        raise KeyError("Dataset configuration not found. Expected 'dataset' or 'datasets' key.")

    # -----------------------------------------------------------------------
    # Load or download full dataset
    # -----------------------------------------------------------------------
    os.makedirs(DATASETS_DIR, exist_ok=True)
    local_path = os.path.join(DATASETS_DIR, dataset_name.replace("/", "_"))

    if os.path.exists(local_path):
        print(f"📂 Loading dataset from local cache: {local_path}")
        dataset = load_from_disk(local_path)
    else:
        print(f"⬇️  Downloading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset.save_to_disk(local_path)
        print(f"✅ Full dataset saved locally to: {local_path}")

    # -----------------------------------------------------------------------
    # Handle variations in split keys and select subsets dynamically
    # -----------------------------------------------------------------------
    val_key = "validation" if "validation" in dataset else "val"

    train = select_subset(dataset["train"], n_train, seed=seed)
    val = select_subset(dataset[val_key], n_val, seed=seed)
    test = select_subset(dataset["test"], n_test, seed=seed)

    print(f"📊 Loaded {len(train)} train / {len(val)} val / {len(test)} test samples (from full cache).")
    return train, val, test


# ---------------------------------------------------------------------------
# Prompt / Message Construction
# ---------------------------------------------------------------------------

def build_user_prompt(dialogue: str, task_instruction: str) -> str:
    """Construct a summarization-style prompt given a dialogue and instruction."""
    return f"{task_instruction}\n\n## Dialogue:\n{dialogue}\n## Summary:"
    

def build_messages_for_sample(sample, task_instruction, include_assistant=False):
    """
    Build a chat-style message list for a given sample, compatible with
    models that use chat templates (like Llama 3).
    """
    messages = [
        {
            "role": "user",
            "content": build_user_prompt(sample["dialogue"], task_instruction),
        }
    ]
    if include_assistant:
        messages.append({"role": "assistant", "content": sample["summary"]})
    return messages

def preprocess_samples(examples, tokenizer, task_instruction, max_length):
    """Tokenize dialogues and apply assistant-only masking for causal LM."""
    input_ids_list, labels_list, attn_masks = [], [], []

    for d, s in zip(examples["dialogue"], examples["summary"]):
        sample = {"dialogue": d, "summary": s}

        # Build chat-style text
        msgs_full = build_messages_for_sample(
            sample, task_instruction, include_assistant=True
        )
        msgs_prompt = build_messages_for_sample(
            sample, task_instruction, include_assistant=False
        )

        text_full = tokenizer.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False
        )
        text_prompt = tokenizer.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(text_prompt)
        tokens = tokenizer(
            text_full,
            max_length=max_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        # Mask non-assistant tokens
        start_idx = len(tokens["input_ids"])
        for i, (start, _) in enumerate(tokens["offset_mapping"]):
            if start >= prompt_len:
                start_idx = i
                break

        labels = [-100] * start_idx + tokens["input_ids"][start_idx:]
        input_ids_list.append(tokens["input_ids"])
        labels_list.append(labels)
        attn_masks.append(tokens["attention_mask"])

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_masks,
    }

def main():

    import pandas as pd
    from pprint import pprint
    from datasets import Dataset
    from transformers import AutoTokenizer
    from dotenv import load_dotenv
    from utils.hf_utils import authenticate_huggingface

    load_dotenv()

    # Authenticate with Hugging Face if token is available
    authenticate_huggingface()

    sample = {
        "dialogue": (
            "A: Hi!\n"
            "B: Hello! How are you?\n"
            "A: I'm great, thanks!"
        ),
        "summary": "A greets B and says they're doing well.",
    }

    task_instruction = (
        "You are a helpful assistant who writes concise, factual summaries of conversations. "
        "Summarize the following conversation into a single sentence. "
    )


    sample_messages_full = build_messages_for_sample(
        sample,
        task_instruction,
        include_assistant=True
    )

    # STEP 2: Convert to chat template
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    chat_full = tokenizer.apply_chat_template(
        sample_messages_full,
        tokenize=False,
        add_generation_prompt=False
    )

    # STEP 3/4: Tokenize and apply assistant-only masking
    samples_dataset = Dataset.from_dict({
        "dialogue": [sample["dialogue"]],
        "summary": [sample["summary"]],
    })
    processed_samples = preprocess_samples(samples_dataset, tokenizer, task_instruction, max_length=256)

    print("=" * 80)
    print("📜 ORIGINAL EXAMPLE:\n")
    pprint(sample)
    print("=" * 80)
    print("📜 STEP 1: AFTER CONVERTING TO 'MESSAGES':\n")
    pprint(sample_messages_full)
    print("=" * 80)
    print("STEP 2: AFTER APPLYING CHAT TEMPLATE: \n")
    print(chat_full)
    print("=" * 80)

    # ---------------------------------------------------------------------------
    # Visualize tokenization and masking
    # ---------------------------------------------------------------------------
    print("STEP 3/4: AFTER TOKENIZATION AND MASKING: \n")

    # Extract first example
    ex = {k: v[0] for k, v in processed_samples.items()}
    tokens = [tokenizer.decode([tid]) for tid in ex["input_ids"]]

    df = pd.DataFrame({
        "token": tokens,
        "input_id": ex["input_ids"],
        "attention_mask": ex["attention_mask"],
        "label": ex["labels"],
    })
    df["masked"] = df["label"].apply(lambda x: x == -100)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    print(df)

if __name__ == "__main__":
    main()