import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from transformers import logging
logging.set_verbosity_error()

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, tokenizer_name, device):
    tokenizer = AutoTokenizer.from_pretrained(f"igemugm/{tokenizer_name}-stress-predictor", do_lower_case=False)
    config = BertConfig.from_pretrained(f"igemugm/{model_name}-stress-predictor")
    model = AutoModelForSequenceClassification.from_pretrained(f"igemugm/{model_name}-stress-predictor", config=config)
    model.to(device)
    model.eval()
    return tokenizer, model

def region_stress_classification(
    model, tokenizer, sequence, device, 
    window_size=200, stride=100, save_path="visualization.png"
): 
    seq_len = len(sequence)
    if seq_len != 1000 and seq_len != 2000:
        raise ValueError("Sequence length must be either 1000 or 2000")
    
    pos_votes = [[] for _ in range(seq_len)]
    results = {}
    all_probs = []

    window_id = 1
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        subseq = sequence[start:end]

        inputs = tokenizer(
            subseq,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=window_size
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs).logits
            probs = torch.softmax(outputs, dim=-1)[0].cpu().tolist()
            pred = int(torch.argmax(outputs, dim=-1)[0].item())

        results[str(window_id)] = {
            "sequence": subseq,
            "label_seq": str(pred),
            "score": probs[pred]
        }
        all_probs.append(probs[pred])

        # update votes per posisi
        for i in range(start, end):
            pos_votes[i].append(pred)

        window_id += 1

    results["final_score"] = sum(all_probs) / len(all_probs)

    # === VISUALISASI ===
    colors = []
    for votes in pos_votes:
        if len(votes) == 0:
            colors.append("white")
        elif all(v == 1 for v in votes):
            colors.append("green")
        elif all(v == 0 for v in votes):
            colors.append("red")
        else:
            colors.append("gray")

    plt.figure(figsize=(15, 2))
    plt.scatter(range(seq_len), [1]*seq_len, c=colors, s=30, marker="s")
    plt.title("Region Stress Classification Visualization")
    plt.yticks([])
    plt.xlabel("Sequence Position")

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return results

def promoter_stress_classification(
    model, tokenizer, sequence, device, 
    slice_size=1000, stride=200, window_size=200, output_dir="outputs"
):
    seq_len = len(sequence)
    if seq_len < 5000 or seq_len > 10000:
        raise ValueError("Sequence length must be between 5000 - 10000")
    elif seq_len % 1000 != 0:
        raise ValueError("Sequence length must be divisible by 1000")

    if slice_size == 1000:
        valid_strides = [100, 200, 500]
    elif slice_size == 2000:
        valid_strides = [100, 200, 400, 500]
    else:
        raise ValueError("Slice size must be either 1000 or 2000")

    if stride not in valid_strides:
        raise ValueError(f"Stride {stride} is not valid for slice {slice_size}. "
                         f"Valid: {valid_strides}")

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    num_slices = (len(sequence) + slice_size - 1) // slice_size

    for slice_id in range(num_slices):
        start = slice_id * slice_size
        end = min((slice_id + 1) * slice_size, len(sequence))
        subseq = sequence[start:end]

        save_path = os.path.join(output_dir, f"slice{slice_id+1}_stride{stride}.png")
        res = region_stress_classification(
            model, tokenizer, subseq, device, 
            window_size=window_size, stride=stride, save_path=save_path
        )
        results[f"slice_{slice_id+1}"] = res

    return results
