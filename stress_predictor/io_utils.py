from Bio import SeqIO
import json

def read_fasta(fasta_path):
    """Read a single sequence from a FASTA file (error if more than one)."""
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if len(records) == 0:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")
    if len(records) > 1:
        raise ValueError(f"FASTA file contains more than one sequence, but only one is allowed: {fasta_path}")
    record = records[0]
    return str(record.seq)


def write_output(results, output_path):
    """
    Write prediction results to a JSON file
    results: dict or list of dicts
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
