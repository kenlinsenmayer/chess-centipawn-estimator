import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "lichess_fen_cp_20M.csv"


def main():
    # Read the CSV file
    df = pd.read_csv(CSV_FILE)

    # Summary statistics
    print("Summary statistics for centipawn (cp) values:")
    print(df['cp_standardized_clipped'].describe())

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['cp_standardized_clipped'], bins=100, color='skyblue', edgecolor='black')
    plt.title('Histogram of Centipawn (cp) Values')
    plt.xlabel('Centipawn (cp)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cp_histogram.png')
    plt.show()

if __name__ == "__main__":
    main()
