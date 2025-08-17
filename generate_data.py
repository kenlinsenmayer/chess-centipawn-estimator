# Script to download, decompress, and process lichess_db_eval.jsonl.zst
# Extracts FEN and centipawn evaluation at highest depth for each position
# Output: one line per position: <FEN> <CP>

import os
import json
import requests
import zstandard as zstd
from tqdm import tqdm

DOWNLOAD_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
ZST_FILE = "lichess_db_eval.jsonl.zst"
JSONL_FILE = "lichess_db_eval.jsonl"
OUTPUT_FILE = "lichess_fen_cp.csv"


def download_file(url, dest):
	if os.path.exists(dest):
		print(f"File {dest} already exists. Skipping download.")
		return
	with requests.get(url, stream=True) as r:
		r.raise_for_status()
		with open(dest, 'wb') as f:
			for chunk in r.iter_content(chunk_size=8192):
				f.write(chunk)
	print(f"Downloaded {dest}")


def decompress_zst(zst_path, out_path):
	if os.path.exists(out_path):
		print(f"File {out_path} already exists. Skipping decompression.")
		return
	file_size = os.path.getsize(zst_path)
	chunk_size = 1024 * 1024  # 1MB
	with open(zst_path, 'rb') as compressed, open(out_path, 'wb') as out:
		dctx = zstd.ZstdDecompressor()
		with tqdm(total=file_size, unit='B', unit_scale=True,
				  desc='Decompressing') as pbar:
			reader = dctx.stream_reader(compressed)
			while True:
				chunk = reader.read(chunk_size)
				if not chunk:
					break
				out.write(chunk)
				pbar.update(len(chunk))
	print(f"Decompressed to {out_path}")


def process_jsonl(jsonl_path, output_path):
	# Estimate total lines for progress bar
	print("Estimating total lines for progress bar...")
	file_size = os.path.getsize(jsonl_path)
	avg_line_length = 180  # bytes, adjust if you have a better estimate
	est_total_lines = max(1, file_size // avg_line_length)
	import csv
	with (
		open(jsonl_path, 'r') as fin,
		open(output_path, 'w', newline='') as fout,
		tqdm(total=est_total_lines, desc='Processing (approx)', unit='lines') as pbar
	):
		writer = csv.writer(fout)
		writer.writerow(["fen", "cp"])
		for line in fin:
			try:
				data = json.loads(line)
				fen = data.get('fen')
				evals = data.get('evals', [])
				if not fen or not evals:
					pbar.update(1)
					continue
				# Find the evaluation with the highest depth
				max_depth_eval = max(evals, key=lambda e: e.get('depth', 0))
				pvs = max_depth_eval.get('pvs', [])
				if not pvs or 'cp' not in pvs[0]:
					pbar.update(1)
					continue
				cp = pvs[0]['cp']
				writer.writerow([fen, cp])
			except Exception:
				# Optionally log or count errors
				pass
			pbar.update(1)
	print(f"Processed and saved output to {output_path}")


if __name__ == "__main__":
	download_file(DOWNLOAD_URL, ZST_FILE)
	decompress_zst(ZST_FILE, JSONL_FILE)
	process_jsonl(JSONL_FILE, OUTPUT_FILE)
