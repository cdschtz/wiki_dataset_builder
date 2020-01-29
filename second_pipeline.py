import os
import json
import time
import subprocess
from pathlib import Path

all_files = []
base_path = "/Users/christopher/Coding/BA_Code/wikiextractor/new_output"


for root, directories, filenames in os.walk(base_path):
     for filename in filenames:
     	if '.' in filename:
     		continue
     	else:
             all_files.append(os.path.join(root, filename))

all_files = sorted(all_files)


if __name__ == '__main__':
	# print(len(all_files))

	entries_counter = 0

	for file in all_files:

		for line in file:

			entries_counter += 1

	print(f'Total articles: {entries_counter}')


