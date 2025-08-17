import sys
import pandas as pd

INPUT_CSV = "lichess_fen_cp_standardized.csv"

def main():
    if len(sys.argv) != 2:
        print("Usage: python subset_csv.py <num_entries>")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
    except ValueError:
        print("Please provide a valid integer for <num_entries>.")
        sys.exit(1)
    output_csv = f"lichess_fen_cp_standardized_{n}.csv"
    df = pd.read_csv(INPUT_CSV, nrows=n)
    df.to_csv(output_csv, index=False)
    print(f"Saved {n} entries to {output_csv}")

if __name__ == "__main__":
    main()
