import configparser
import glob
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import global_config
from CGRU import CGRU
from CustomDataLoader import CustomDataset

cfg = configparser.ConfigParser()
cfg.read(global_config.model_config)

train_dir = global_config.train_dir
val_dir = global_config.test_dir
MODEL_PATH = 'model/final_u.pt'

image_fn = [os.path.basename(x).split('.')[0].lower() for x in glob.glob(os.path.join(train_dir ,'*.png'))]
image_fn = "".join(image_fn)
letters = sorted(list(set(list(image_fn))))
print(len(letters))
print(letters)
# print(image_fn)


vocabulary = ["-"] + letters
print('voc', len(vocabulary))
print(vocabulary)
idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
print(idx2char)

char2idx = {v: k for k, v in idx2char.items()}
print(char2idx)

number_chars = len(char2idx)
print(number_chars)


def decode_predictions(text_batch_logits):
    global idx2char

    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2)  # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T  # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new


val_dataset = CustomDataset(val_dir)

val_dataloader = DataLoader(val_dataset, batch_size=4,
                            num_workers=4, shuffle=False)

results_test = pd.DataFrame(columns=['actual', 'prediction'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

cgru = CGRU(number_chars)
cgru.load_state_dict(torch.load(MODEL_PATH, map_location=device))

print('model loaded..............................')

with torch.no_grad():
    for image_batch, text_batch in tqdm(val_dataloader, leave=True):
        text_batch_logits = cgru(image_batch.to(device))  # [T, batch_size, num_classes==num_features]
        text_batch_pred = decode_predictions(text_batch_logits.cpu())
        # print(text_batch, text_batch_pred)
        df = pd.DataFrame(columns=['actual', 'prediction'])
        df['actual'] = text_batch
        df['prediction'] = text_batch_pred
        results_test = pd.concat([results_test, df])
results_test = results_test.reset_index(drop=True)

print(results_test.shape)
print(results_test.head())
