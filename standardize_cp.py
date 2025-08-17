import pandas as pd

INPUT_CSV = "lichess_fen_cp.csv"
OUTPUT_CSV = "lichess_fen_cp_standardized.csv"

def main():
    # Load the CSV
    df = pd.read_csv(INPUT_CSV)

    # Remove entries with cp == 0
    df = df[df['cp'] != 0]

    # Standardize cp (z-score)
    mean_cp = df['cp'].mean()
    std_cp = df['cp'].std()
    df['cp_standardized'] = (df['cp'] - mean_cp) / std_cp

    # Save to new CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} entries to {OUTPUT_CSV}")
    print(df.head())

if __name__ == "__main__":
    main()
