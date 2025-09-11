import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Seq Mutator CLI with Hugging Face model")
    parser.add_argument("fasta", type=str, help="Input FASTA file")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--tokenizer", type=str, required=True, help="Hugging Face tokenizer name or path")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file (CSV)")
    parser.add_argument("--max-length", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--force-cpu", action="store_true", help="Force using CPU even if GPU is available")

    # mutually exclusive group: usermust choose one (and only one)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pr", action="store_true", help="Enable promoter detection mode")
    group.add_argument("--rg", action="store_true", help="Enable region-only mode")
    
    return parser.parse_args()
