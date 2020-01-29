import os

def count_articles(path='/Users/christopher/Coding/BA_Code/wiki_dataset_builder/new_output'):
    import time
    start = time.time()
    article_counter = 0
    all_files = []
    for root, directories, filenames in os.walk(path):
     for filename in filenames:
     	if '.' in filename:
     		continue
     	else:
             all_files.append(os.path.join(root, filename))
    
    for file in all_files:
        for line in file:
            article_counter += 1

    elapsed = time.time() - start
    print('Finished counting articles in {:4.2f} seconds'.format(elapsed))
    print(f'Total articles: {article_counter}')


if __name__ == '__main__':
	count_articles()
