import pandas as pd
import sys

INPUT_CSV = "lichess_fen_cp_standardized_20000000.csv"
OUTPUT_CSV = "lichess_fen_cp_standardized_20000000_clipped_-5_5.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    df['cp_standardized_clipped'] = df['cp_standardized'].clip(-5, 5)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved clipped file to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
