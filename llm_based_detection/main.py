import argparse
import os
import sys

from pathlib import Path
from dotenv import load_dotenv

from llm_judge import evaluate_jsonl

def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="Hallucination Judge for LLM responses")
    p.add_argument("--input", default="product_info.jsonl", 
                   help="Input JSONL file with rows containing id, question, context, response")
    p.add_argument("--provider", choices=["openai", "anthropic"], default="openai",
                   help="LLM provider to use for judging")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="Model name to use")
    p.add_argument("--tau", type=float, default=0.80,
                   help="Threshold for groundedness decision (0.0-1.0)")
    p.add_argument("--out-dir", default="./judge_outputs",
                   help="Output directory for results")
    p.add_argument("--max-rows", type=int, default=0,
                   help="Maximum number of rows to process (0 = all)")
    p.add_argument("--timeout", type=int, default=60,
                   help="Timeout for LLM calls in seconds")
    args = p.parse_args()

    # Validate tau parameter
    if not (0.0 <= args.tau <= 1.0):
        print(f"Error: tau must be between 0.0 and 1.0, got {args.tau}", file=sys.stderr)
        sys.exit(1)

    # Check API key & arguments
    if args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: Missing OPENAI_API_KEY in environment variables", file=sys.stderr)
            print("Please set your OpenAI API key in a .env file or environment variable", file=sys.stderr)
            sys.exit(1)
    elif args.provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("Error: Missing ANTHROPIC_API_KEY in environment variables", file=sys.stderr)
            print("Please set your Anthropic API key in a .env file or environment variable", file=sys.stderr)
            sys.exit(1)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file not found: {in_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    if not in_path.is_file():
        print(f"Error: Input path is not a file: {in_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"Starting evaluation...")
    print(f"  Input: {in_path.resolve()}")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Tau (threshold): {args.tau}")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Max rows: {'all' if args.max_rows == 0 else args.max_rows}")
    print(f"  Timeout: {args.timeout}s")
    print()

    try:
        results, processing_stats = evaluate_jsonl(
            input_path=str(in_path),
            provider=args.provider,
            model=args.model,
            tau=args.tau,
            out_dir=args.out_dir,
            max_rows=args.max_rows,
            timeout=args.timeout,
        )

        # Print processing statistics first
        print(f"Processing Statistics:")
        print(f"  Total lines in input: {processing_stats.total_lines}")
        print(f"  Empty lines: {processing_stats.empty_lines}")
        print(f"  Invalid JSON: {processing_stats.invalid_json}")
        print(f"  Missing required fields: {processing_stats.missing_fields}")
        print(f"  Empty required fields: {processing_stats.empty_fields}")
        print(f"  Processing errors: {processing_stats.processing_errors}")
        print(f"  Successfully processed: {processing_stats.successfully_processed}")
        
        skipped_total = (processing_stats.empty_lines + processing_stats.invalid_json + 
                        processing_stats.missing_fields + processing_stats.empty_fields + 
                        processing_stats.processing_errors)
        print(f"  Total skipped: {skipped_total}")
        
        if skipped_total > 0:
            print(f"  Data integrity: {processing_stats.successfully_processed}/{processing_stats.total_lines - processing_stats.empty_lines} valid rows processed ({100 * processing_stats.successfully_processed / (processing_stats.total_lines - processing_stats.empty_lines):.1f}%)")
        
        print()

        if results:
            print(f"Successfully processed {len(results)} rows")
            print(f"Results summary:")
            
            # Count decisions
            fact_count = sum(1 for r in results if r.decision == "FACT")
            hallucination_count = sum(1 for r in results if r.decision == "HALLUCINATION")
            unknown_count = sum(1 for r in results if r.decision == "UNKNOWN")
            
            print(f"  - FACT: {fact_count}")
            print(f"  - HALLUCINATION: {hallucination_count}")
            print(f"  - UNKNOWN: {unknown_count}")
            
            # Average groundedness
            if results:
                avg_groundedness = sum(r.overall_groundedness for r in results) / len(results)
                print(f"  - Average groundedness: {avg_groundedness:.4f}")
            
            print(f"Output files written to: {args.out_dir}")
            print(f"  - Detailed results: {args.out_dir}/results.jsonl")
            print(f"  - Summary CSV: {args.out_dir}/summary.csv")
            print(f"  - Processing log: {args.out_dir}/processing_log.jsonl")
            print(f"  - Processing summary: {args.out_dir}/processing_summary.json")
            print(f"  - Traces: {args.out_dir}/traces/")
            
            print(f"\nData Verification:")
            print(f"  Each result in results.jsonl includes 'input_row' and 'line_number' fields")
            print(f"  All skipped rows are logged in processing_log.jsonl with reasons")
            print(f"  Processing summary contains complete audit trail")
            
            print(f"\nSample result:")
            sample = results[0]
            print(f"  ID: {sample.id}")
            print(f"  Decision: {sample.decision}")
            print(f"  Groundedness: {sample.overall_groundedness:.4f}")
            print(f"  Sentences: {len(sample.sentences)}")
            if sample.sentences:
                print(f"  First sentence: {sample.sentences[0][:100]}{'...' if len(sample.sentences[0]) > 100 else ''}")
        else:
            print("No results returned. Check your input file format.", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()