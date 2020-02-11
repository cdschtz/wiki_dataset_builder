import os
import json


def count_articles(path='/Users/christopher/Coding/BA_Code/wiki_dataset_builder/data/output_before_gen/'):
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
    print('INFO: Finished counting articles in {:4.2f} seconds'.format(elapsed))
    print(f'Total articles: {article_counter}')


def display_text_of_articles(path='/Users/christopher/Coding/BA_Code/wiki_dataset_builder/generation_examples/wiki_00'):
    with open(path, 'r') as r_file:
        with open('text_display/' + path[-7:], 'w') as w_file:
            for line in r_file:
                tmp = json.loads(line)
                # w_file.write(json.dumps(str(tmp['text'])) + '\n\n\n\n\n')
                # print(tmp['text'])
                print(tmp['text'] + '\n' * 10, file=w_file)


def get_original_text(idx, or_file_path):
    # TODO: remember to parse the beginning until '\n\n\
    path = or_file_path
    with open(path, 'r') as r_file:
        for i, line in enumerate(r_file):
            if i == idx:
                # tmp['text'].split('\n\n', 1)[1][:in_len]
                return json.loads(line)['text'].split('\n\n', 1)[1]
                # return json.loads(line)['text']


def visualize_text_of_file(f_path):

    f_vis_obj = {
        "combination": {
            "in_len": 0,
            "out_len": 0,
            "temp": 0,
            "rep_pen": 0,
            "bs_size": 0,
        },
        # "info": {},
        "articles": []
    }

    with open(f_path, 'r') as r_file:
        # TODO: parse path for combination values
        # f_path ex: wiki_00_i_40_o_100_nb_5_t_1.0_rp_1.0.jsonl
        split_path = f_path.split('/')[-1].split('_')
        f_vis_obj['in_len'] = int(split_path[3])
        f_vis_obj['out_len'] = split_path[5]
        f_vis_obj['bs_size'] = split_path[7]
        f_vis_obj['temp'] = split_path[9]
        f_vis_obj['rep_pen'] = split_path[11][:-6]
        
        tmp = f_path.split('/')
        or_folder = '/'.join(tmp[:6]) + '/output/partition00/AA/'
        or_file = '_'.join(split_path[:2])

        for i, line in enumerate(r_file):
            or_text = get_original_text(i, or_folder + or_file)
            tmp = json.loads(line)
            title = tmp['title']
            gen_text = tmp['text']
            art_obj = {
                "art_num": i + 1,
                "title": title,
                "input": or_text[:f_vis_obj['in_len']],
                "original": or_text[:5*int(f_vis_obj['out_len'])],
                "generated": gen_text
            }
            f_vis_obj['articles'].append(art_obj)
    
    out_path = f_path[:-6] + '.txt'
    with open(out_path, 'w') as w_file:
        w_file.write('Combination:\n')
        w_file.write('- Input Length: ' + str(f_vis_obj['in_len']) + ' chars\n')
        w_file.write('- Output Length: ' + f_vis_obj['out_len'] + ' tokens\n')
        w_file.write('- Temperature: ' + f_vis_obj['temp']+ '\n')
        w_file.write('- Repetition Penalty: ' + f_vis_obj['rep_pen']+ '\n')
        w_file.write('- Beam Search Size: ' + f_vis_obj['bs_size'])
        for article in f_vis_obj['articles']:
            w_file.write('\n\n\n\n\n' + 40 * '#' + ' ARTICLE ' + str(article['art_num']) + ' ' + 40 * '#')
            w_file.write('\n' * 3 + 'Title: ' + article['title'])
            w_file.write('\n' * 2 + 'Input: ' + article['input'])
            w_file.write('\n' * 2 + 'ORIGINAL ' + 30 * '=')
            w_file.write('\n' + article['original'])
            w_file.write('\n' * 4 + 'GENERATED ' + 30 * '=')
            w_file.write('\n' + article['generated'][:-1])


def visualize_text_of_folder(folder_path):
    all_files = []
    all_files = os.listdir('./generation_examples/')

    all_files = sorted(all_files)

    for file in all_files:
        if '.txt' not in file and 'DS_Store' not in file:
            visualize_text_of_file(folder_path + file)


def to_word_splitter(txt):
    print(txt[::-1].split(' ', 1)[1][::-1])
    pass

if __name__ == '__main__':
    # count_articles()
    # display_text_of_articles()
    # visualize_text_of_file('/Users/christopher/Coding/BA_Code/wiki_dataset_builder/generation_examples/wiki_00_i_40_o_100_nb_5_t_1.0_rp_1.0.jsonl')
    visualize_text_of_folder('/Users/christopher/Coding/BA_Code/wiki_dataset_builder/generation_examples/')
    # to_word_splitter('Animalia is an ')
