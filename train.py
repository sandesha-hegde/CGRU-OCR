import os
import glob
import torch
import numpy as np
import torch.nn as nn
from CustomDataLoader import CustomDataset
from CGRU import CGRU


import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader


import configparser
import global_config

cfg = configparser.ConfigParser()
cfg.read(global_config.model_config)

train_dir = global_config.train_dir
val_dir = global_config.test_dir
MODEL_DIR = global_config.model_dir


image_fn = [os.path.basename(x).split('.')[0].lower() for x in glob.glob(os.path.join(train_dir, '*.png'))]
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


def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


train_dataset = CustomDataset(train_dir)
val_dataset = CustomDataset(val_dir)

train_dataloader = DataLoader(train_dataset, batch_size=4,
                              num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4,
                            num_workers=4, shuffle=False)

imgs, texts = iter(val_dataloader).next()
print(imgs.shape, len(texts))

print(texts)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def encode_text_batch(text_batch):
    global char2idx, idx2char
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)

    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)

    return text_batch_targets, text_batch_targets_lens


criterion = nn.CTCLoss(blank=0)


def compute_loss(text_batch, text_batch_logits):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    text_batch_logps = F.log_softmax(text_batch_logits, 2)  # [T, batch_size, num_classes]
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),),
                                       fill_value=text_batch_logps.size(0),
                                       dtype=torch.int32).to(device)  # [batch_size]
    # print(text_batch_logps.shape)
    # print(text_batch_logps_lens)
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch)
    # print(text_batch_targets)
    # print(text_batch_targets_lens)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss


def decode_predictions(text_batch_logits):
    text_batch_tokens = F.softmax(text_batch_logits, 2).argmax(2)  # [T, batch_size]
    text_batch_tokens = text_batch_tokens.numpy().T  # [batch_size, T]

    text_batch_tokens_new = []
    for text_tokens in text_batch_tokens:
        text = [idx2char[idx] for idx in text_tokens]
        text = "".join(text)
        text_batch_tokens_new.append(text)

    return text_batch_tokens_new


# def acc_calc(model, dataset, label_converter) -> float:
#     acc = 0
#     with torch.no_grad():
#         model.eval()
#         for idx in range(len(dataset)):
#             img, text = dataset[idx]
#             logits = model(img.unsqueeze(0).to(device))
#             pred_text = decode_prediction(logits.cpu(), label_converter)

#             if pred_text == text:
#                 acc += 1

#     return acc / len(dataset)


def validation(model, val_dataloader):
    val_losses = []
    with torch.no_grad():
        model.eval()
        for batch_img, batch_text in val_dataloader:
            logits = cgru(batch_img.to(device))
            val_loss = compute_loss(batch_text, logits)
            val_losses.append(val_loss.item())
    return val_losses


print(encode_text_batch(text_batch=texts))

number_chars = len(char2idx)
print(number_chars)

# crnn = CRNN(number_chars)
# crnn.apply(weights_init)
# crnn = crnn.to(device)

# text_batch_logits = crnn(imgs.to(device))
# print(texts)
# print(text_batch_logits.shape)


# print(compute_loss(texts, text_batch_logits))

num_epochs = 50
lr = 0.001
weight_decay = 1e-3
clip_norm = 5

cgru = CGRU(number_chars)
cgru.apply(weights_init)
cgru = cgru.to(device)

optimizer = optim.Adam(cgru.parameters(), lr=lr, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)

epoch_losses = []
iteration_losses = []
num_updates_epochs = []

try:
    for epoch in tqdm(range(1, num_epochs + 1)):
        epoch_loss_list = []
        num_updates_epoch = 0
        for image_batch, text_batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            text_batch_logits = cgru(image_batch.to(device))
            loss = compute_loss(text_batch, text_batch_logits)
            iteration_loss = loss.item()

            if np.isnan(iteration_loss) or np.isinf(iteration_loss):
                continue

            num_updates_epoch += 1
            iteration_losses.append(iteration_loss)
            epoch_loss_list.append(iteration_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(cgru.parameters(), clip_norm)
            optimizer.step()

        epoch_loss = np.mean(epoch_loss_list)
        print("Epoch:{}    Loss:{}    NumUpdates:{}".format(epoch, epoch_loss, num_updates_epoch))
        epoch_losses.append(epoch_loss)
        num_updates_epochs.append(num_updates_epoch)
        lr_scheduler.step(epoch_loss)

        print('Validation...........................................')

        val_losses = validation(cgru, val_dataloader)

        val_loss = np.mean(val_losses)

        print("Validation Val_Loss:{}".format(val_loss))
except KeyboardInterrupt:

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    torch.save(cgru.state_dict(), os.path.join(MODEL_DIR, 'model.pt'))

torch.save(cgru.state_dict(), os.path.join(MODEL_DIR, 'final.pt'))


