import argparse
import sys
import json

# Import modules
from ingest import run_ingestion
from retrieve import generate_character_info

def main():
    parser = argparse.ArgumentParser(description="Story RAG Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command 1: compute-embeddings
    parser_compute = subparsers.add_parser("compute-embeddings", help="Process stories and create vector db")

    # Command 2: get-character-info
    parser_info = subparsers.add_parser("get-character-info", help="Get details about a specific character")
    parser_info.add_argument("name", type=str, help="The name of the character to search for")
    parser_info.add_argument("--verbose", action="store_true", help="Show raw LLM output for debugging")

    args = parser.parse_args()

    if args.command == "compute-embeddings":
        try:
            run_ingestion()
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred: {e}")

    elif args.command == "get-character-info":
        print(f"🔎 Searching for character: {args.name}...")
        result = generate_character_info(args.name, verbose=args.verbose)
        
        # Pretty print the final JSON result
        print("\n" + json.dumps(result, indent=4))

    else:
        parser.print_help()

if __name__ == "__main__":
    main()