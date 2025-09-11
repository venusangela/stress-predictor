import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.configuration_bert import BertConfig
import math
from collections import Counter

from transformers import logging
logging.set_verbosity_error()

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name, tokenizer_name, device):
    tokenizer = AutoTokenizer.from_pretrained(f"models/{tokenizer_name}", do_lower_case=False)
    config = BertConfig.from_pretrained(f"models/{model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(f"models/{model_name}", config=config)
    model.to(device)
    model.eval()
    return tokenizer, model

def slice_sequence(seq, window_size=200):
    seq_len = len(seq)
    n_windows = math.ceil(seq_len / window_size)

    if seq_len % window_size == 0:
        stride = window_size
    else:
        stride = (seq_len - window_size) // (n_windows - 1)

    slices = []
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        if end > seq_len:
            end = seq_len
            start = end - window_size

        slices.append(
            {
                "sequence": seq[start:end],
                "start": start,
                "end": end
            }
        )

    return slices

def promoter_stress_classification(model, tokenizer, sequence, device, max_length=200):
    seqs_info = slice_sequence(sequence, max_length)
    seqs = [s["sequence"] for s in seqs_info]
    inputs = tokenizer(seqs, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
    
    pred_score = sum(pred) / len(pred)
    final_label = "stress promoter" if pred_score > 0.5 else "non-stress promoter"
    result = {
        "slices": [
            {"start_index": s["start"], "end_index": s["end"], "sequence": s["sequence"], "pred": pred[i]}
            for i,s in enumerate(seqs_info)
        ],
        "pred_score": pred_score,
        "final_label": final_label
    }
    return result

def region_stress_classification(model, tokenizer, sequence, device, window_size=200, stride=1):
    seq_len = len(sequence)
    pos_votes = [[] for _ in range(seq_len)]

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
            pred = torch.argmax(outputs, dim=-1).item()

        for i in range(start, end):
            pos_votes[i].append(pred)

    final_pos_labels = []
    for votes in pos_votes:
        if len(votes) == 0:
            final_pos_labels.append("0")
        else:
            c = Counter(votes)
            label = "1" if c[1] >= c[0] else "0"
            final_pos_labels.append(label)

    final_label_seq = "".join(final_pos_labels)

    result = {
        "promoter_sequence": sequence,
        "final_label_seq": final_label_seq
    }

    return result

