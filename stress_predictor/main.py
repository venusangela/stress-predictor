from stress_predictor import read_fasta, write_output, load_model, get_device, get_args
from stress_predictor import promoter_stress_classification, region_stress_classification

def main():
    args = get_args()

    # Device
    device = get_device(args.force_cpu)
    print(f"Using device: {device}")

    # Load model
    tokenizer, model = load_model(args.model, args.tokenizer, device)

    # Read sequences
    sequence = read_fasta(args.fasta)

    # Process
    if args.pr:
        result = promoter_stress_classification(model, tokenizer, sequence, device, max_length=args.max_length)
    elif args.rg:
        result = region_stress_classification(model, tokenizer, sequence, device)

    # Write output
    write_output(result, args.output)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()