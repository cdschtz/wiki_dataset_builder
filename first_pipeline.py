import os
import json
import time
import subprocess
from pathlib import Path

all_files = []
wiki_ectractor_output_path = "/Users/christopher/Coding/BA_Code/wikiextractor/new_output"
output_base_path = '/Users/christopher/Coding/BA_Code/wiki_dataset_builder/new_output'
for root, directories, filenames in os.walk(wiki_ectractor_output_path):
     for filename in filenames:
     	if '.' in filename:
     		continue
     	else:
             all_files.append(os.path.join(root, filename))

all_files = sorted(all_files)
# print(*all_files, sep='\n')

def extract_and_shorten(path):
	new_lines = []
	with open(path, 'r') as file:
		for i, line in enumerate(file):
			res = json.loads(line)
			res['text'] = res['text'][:1000]

			json_tmp = json.dumps(res)
			new_lines.append(json_tmp + '\n')

	Path(output_base_path + '/' + path[55:-7]).mkdir(parents=True, exist_ok=True)
	with open(output_base_path + '/' + path[55:], 'w') as file:

		for line in new_lines:
			file.write(line)



if __name__ == '__main__':

	start_time = time.time()

	from multiprocessing import Pool

	paths = all_files

	pool = Pool(processes = 2)
	# pool = Pool(processes=1)
	pool.map(extract_and_shorten, paths)

	elapsed_time = time.time() - start_time
	print('Elapsed Time: {:06.2f}'.format(elapsed_time))

