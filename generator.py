#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import time
import json
import torch
import logging
from colorama import Fore, Style
from datetime import datetime
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# In[3]:


from generator_factory import generate


# In[4]:


logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
model_path = './model_files/gpt2-xl/'


# In[5]:


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', output_loading_info=False)


# In[ ]:


model = GPT2LMHeadModel.from_pretrained('gpt2-medium')


# In[ ]:

if torch.cuda.is_available():
    print('CUDA device is available')
    model.to('cuda')
else:
    print('CUDA device missing')
model.eval()

# In[96]:


def gen_text_from_text(input_text, config, verbose=False):
    start_time = time.time()
    
    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
    if torch.cuda.is_available():
        # input_ids.to('cuda')
        input_ids = input_ids.cuda()

    with torch.no_grad():
        outputs = generate(
            model,
            input_ids=input_ids,
            max_length=config['output_length'],
            do_sample=True,
            num_beams=config['num_beams'],
            temperature=config['temperature'],
            top_k=50,
            top_p=1.0,
            repetition_penalty=config['repetition_penalty'],
            bos_token_id=0,
            pad_token_id=0,
            eos_token_ids=[0],
            length_penalty=1.0,
            num_return_sequences=config['num_return_sequences']
        )
    
    elapsed_time = time.time() - start_time
    if verbose:
        logging.info(' Finished Text From Text Generation in {:4.2f} seconds.'.format(elapsed_time))
    
    if config['num_return_sequences'] == 1:
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        pass
#         for i in range(config_num_return_sequences):
#             print('Generated {}: {}'.format(i, tokenizer.decode(outputs[0][i], skip_special_tokens=True)))


# In[95]:


def gen_text_from_file(in_path, config, out_path=None, verbose=False):
    original_file_articles = []
    generated_texts = []
    input_texts = []
    
    with open(in_path, 'r') as r_file:
        for i, line in enumerate(r_file):
            json_tmp = json.loads(line)
            
            # split the first part until \n\n, which is the title and only use the
            # amount of characters given by input length
            input_text = json_tmp['text'].split('\n\n', 1)[1][:config['input_length']]
            # make sure that input text is word split (not character split) as this
            # generates higher quality output
            input_text = input_text[::-1].split(' ', 1)[1][::-1]
            input_texts.append(input_text)
            
            generated_text = gen_text_from_text(input_text, config)
            generated_texts.append(generated_text)
            
            # original input is temporarily saved to later be written into the finally formatted files
            original_file_articles.append(json_tmp)

    if not out_path:
        return generated_texts
    
    with open(out_path, 'w') as w_file:
        for i, article in enumerate(original_file_articles):
            article.pop('url', None)
            max_text_size = min(len(generated_texts[i]), len(article['text'])) - 1  # -1 to suppress '!' output at end
            
            metadata = {
                "id": article['id'],
                "input": input_texts[i]
            }
            ff_gpt2 = {
                "meta": metadata,
                "label": 1,  # machine,
                "title": article['title'],
                "text": generated_texts[i][:max_text_size]
            }
            ff_human = {
                "meta": metadata,
                "label": 0,  # human
                "title": article['title'],
                "text": article['text'].split('\n\n', 1)[1][:max_text_size]
            }
            
            w_file.write(json.dumps(ff_human) + '\n')
            w_file.write(json.dumps(ff_gpt2) + '\n')


# In[86]:


def generate_text_for_folder(in_path, config, file_range=range(1), verbose=False):
    all_files = []
    # get all files (also in subdirs)
    for root, directories, filenames in os.walk(in_path):
        for filename in filenames:
            if '.' in filename:
                continue
            else:
                all_files.append(os.path.join(root, filename))

    all_files = sorted(all_files)
    total_files = len(all_files[0])
    
    info_string = '{:5}|{:50}|{:10}|{:20}|{:6.2f}'
    if verbose:
        logging.info('TOTAL FILES', total_files)
        logging.info(f'FILES TO GENERATE: {file_range[0]} - {file_range[-1]}')
        logging.info('{:5}|{:50}|{:10}|{:20}|{:9}'.format('No.', 'Filename', 'Status', 'Time', 'Elapsed (seconds)'))

    for i, file_path in enumerate(all_files):
        from pathlib import Path
        # get output path of file
        split_path = file_path.split('/')
        split_path[2] = 'output_after_gen'
        out_path = '/'.join(split_path) + '.jsonl'

        if i in file_range:
            if verbose:
                start_time = time.time()
                current_time = datetime.now().strftime("%H:%M:%S")  # get time in hours:minutes:seconds
                logging.info(info_string.format(i, file_path, 'Started', current_time, 0.00))
            Path(out_path[:-13]).mkdir(parents=True, exist_ok=True)
            gen_text_from_file(file_path, config, out_path, verbose=True)
            if verbose:
                elapsed_time = time.time() - start_time
                current_time = datetime.now().strftime("%H:%M:%S")  # get time in hours:minutes:seconds
                logging.info(info_string.format(i, file_path, 'Finished', current_time, elapsed_time))


# In[87]:


input_folder = './data/output_before_gen/'


# In[88]:


config = {
    'input_length': 60,  # in characters
    'output_length': 50,  # in tokens
    'num_beams': 5,
    'temperature': 1.0,
    'repetition_penalty': 1.3,
    'num_return_sequences': 1
}


# In[94]:


logging.getLogger().setLevel(logging.INFO)
generate_text_for_folder(input_folder, config, file_range=range(100), verbose=True)


# In[42]:


# print(len(af))
# print(*af[:5], sep='\n')

