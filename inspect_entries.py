import json
import pandas as pd

JSONL_FILE = "lichess_db_eval.jsonl"
CSV_FILE = "lichess_fen_cp.csv"

def display_first_jsonl_entries(jsonl_path, n=5):
    print(f"First {n} entries from {jsonl_path}:")
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            data = json.loads(line)
            fen = data.get('fen')
            evals = data.get('evals', [])
            if evals:
                max_depth_eval = max(evals, key=lambda e: e.get('depth', 0))
                pvs = max_depth_eval.get('pvs', [])
                cp = pvs[0]['cp'] if pvs and 'cp' in pvs[0] else None
            else:
                cp = None
            print(f"Entry {i+1}: FEN={fen}, cp={cp}, evals={evals}")
    print()

def display_first_csv_entries(csv_path, n=5):
    print(f"First {n} entries from {csv_path}:")
    df = pd.read_csv(csv_path)
    print(df.head(n))
    print()

def main():
    display_first_jsonl_entries(JSONL_FILE, 5)
    display_first_csv_entries(CSV_FILE, 5)

if __name__ == "__main__":
    main()
