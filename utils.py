import pickle

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
tqdm.pandas()


def convert_lines(df, vocab, bpe, max_sequence_length):
    # Initial output matrix
    outputs = np.zeros((len(df), max_sequence_length))
    
    cls_id = 0
    pad_id = 1
    eos_id = 2

    for (idx, text, _, _) in tqdm(df.itertuples(name=None), total=len(df)):
        subwords = bpe.encode('<s> '+ text +' </s>')
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))

        outputs[idx, :] = np.array(input_ids)
    return outputs

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat), f1_score(labels_flat, pred_flat)

def save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj