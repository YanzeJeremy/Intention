import os
import json
import csv
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from torch.nn import DataParallel

import time
from tqdm import tqdm


# def load_model(base_model_name, adapter_path, tokenizer_path, device='cpu'):
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#     model = AutoModelForCausalLM.from_pretrained(base_model_name)
#     model = PeftModel.from_pretrained(model, adapter_path)
#     if torch.cuda.device_count() > 1:
#         model = DataParallel(model)
#     model.to(device)
    
#     return tokenizer, model


# def generate_text(prompt, tokenizer, model, device='cpu'):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
#     model.eval()
#     with torch.no_grad():
#         output_ids = model.module.generate(input_ids, max_length=256) if isinstance(model, DataParallel) else model.generate(input_ids, max_length=256)

#     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
#     return generated_text

def load_model(base_model_name, adapter_path, tokenizer_path, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Set the padding token ID to avoid warnings
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    model.to(device)
    
    return tokenizer, model


def generate_text(prompt, tokenizer, model, device='cpu'):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    
    model.eval()
    with torch.no_grad():
        output_ids = model.module.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512) \
            if isinstance(model, DataParallel) else model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512)

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text



if __name__ == "__main__":
    overall_time = time.time()
    base_model_name = "meta-llama/Meta-Llama-3.1-8B"
    adapter_path = '../output/V1/checkpoint-1260'
    tokenizer_path = '../output/V1/checkpoint-1260'
    data_path = '../merged_casino_dataset.json'
    split = 'test'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer, model = load_model(base_model_name, adapter_path, tokenizer_path, device)

    with open(data_path) as f:
        data = json.load(f)[split]
    
    with open('llama3_test.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(('id', 'prompt', 'desired_output', 'model_output'))
    for sample in tqdm(data):
        print("--------------------")
        print(sample)
        index = sample['text'].find('[/INST] ')
        model_input = sample['text'][:index+8]
        print(model_input)
        desired_output = sample['text'][index+8:-4]
        print(desired_output)
        # index = model_input.find('<</SYS>>\n\n')
        # prompt = model_input[index+10:-8]
        generated_text = generate_text(model_input, tokenizer, model, device)
        print('---------------------------------')
        print(generated_text)
        # index = generated_text.find('[/INST] ')
        # model_output = generated_text[index+8:]
        with open('llama3_test.csv', 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow((sample['id'], model_input, desired_output, generated_text))

    # print('Overall time', time.time() - overall_time)