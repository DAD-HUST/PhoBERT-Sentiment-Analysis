import os  
import argparse
import warnings

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
from transformers import *
from transformers.modeling_utils import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from utils import *
from models import *
from tqdm import tqdm

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='./data/data.csv')
parser.add_argument('--dict_path', type=str, default="./phobert/dict.txt")
parser.add_argument('--config_path', type=str, default="./phobert/config.json")
parser.add_argument('--rdrsegmenter_path', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default='./phobert/model.bin')
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=69)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--checkpoint_path', type=str, default='./models')
parser.add_argument('--bpe-codes', default="./phobert/bpe.codes", type=str, help='path to fastBPE BPE')


args = parser.parse_args()
bpe = fastBPE(args)
rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')

seed_everything(args.seed)
if torch.cuda.is_available():
    print('Using GPU')
    device = torch.device('cuda')
else:
    print('Using CPU')
    device = torch.device('cpu')

# Load model
config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_labels=1
)

model = RobertaVN.from_pretrained(args.pretrained_path, config=config)
model.eval()
model.to(device)

if torch.cuda.device_count():
    print(f"Training using {torch.cuda.device_count()} gpus")
    model = nn.DataParallel(model)
    tsfm = model.module.roberta
else:
    tsfm = model.roberta


# Load the Dictionary
vocab = Dictionary()
vocab.add_from_file(args.dict_path)

# Load data
train_df = pd.read_csv(args.train_path)
# Preprocessing
train_df['number_of_words'] = train_df['text'].apply(lambda x: len(str(x).strip().split()))
no_text = train_df[train_df['number_of_words'] == 0]
train_df.drop(no_text.index,inplace=True)
no_text = train_df[train_df['text'] == 'content']
train_df.drop(no_text.index,inplace=True)
no_text = train_df[train_df['text'].isnull()]
train_df.drop(no_text.index, inplace=True)
no_text = train_df[train_df['number_of_words'] == 1]
train_df.drop(no_text.index,inplace=True)
# Tokenize
train_df["text"] = train_df["text"].progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
X_train = convert_lines(train_df, vocab, bpe, args.max_sequence_length)
# Label encoder
lb = LabelEncoder()
y = train_df.label.values
y = lb.fit_transform(y)


base_path = "/".join(args.train_path.split("/")[:-1])
save_pkl(os.path.join(base_path, "X_train_raw.pkl"))
save_pkl(os.path.join(base_path, "y.pkl"))

X_train = load_pkl(os.path.join(base_path, "X_train_raw.pkl"))
y = load_pkl(os.path.join(base_path, "y.pkl"))

# # Optimizer and lr schedulers
# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# num_train_optimization_steps = int(args.epochs * len(train_df) / args.batch_size / args.accumulation_steps)
# optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
# # Warmup 100 step before start learning
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)
# scheduler0 = get_constant_schedule(optimizer)

# # Checkpoint path 
# if not os.path.exists(args.checkpoint_path):
#     os.mkdir(args.checkpoint_path)

# splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=32).split(X_train, y))
# for fold, (train_idx, val_idx) in enumerate(splits):
#     print("Training for fold {}".format(fold))
#     best_score = 0
#     if fold != args.fold:
#         continue
#     train_dataset = TensorDataset(torch.tensor(X_train[train_idx], dtype=torch.long), torch.tensor(y[train_idx], dtype=torch.long))
#     valid_dataset = TensorDataset(torch.tensor(X_train[val_idx], dtype=torch.long), torch.tensor(y[val_idx], dtype=torch.long))
#     for child in tsfm.children():
#         for param in child.parameters():
#             if not param.requires_grad():
#                 print("whoopsies")
#             param.requires_grad = False
    
#     frozen = True
#     for epoch in tqdm(range(args.epochs + 1), desc="Training epoch..."):
#         if epoch > 0 and frozen:
#             for child in tsfm.children():
#                 for param in child.parameters():
#                     param.requires_grad = True
#             frozen = False
#             del scheduler0
#             torch.cuda.empty_cache()
        
#         val_preds = 0
#         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
#         valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
#         avg_loss = 0
#         avg_accuracy = 0

#         optimizer.zero_grad()
#         process_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc="Training...")
#         for step, (x_batch, y_batch) in process_bar:
#             model.train()
#             x_batch = x_batch.to(device)
#             y_batch = y_batch.to(device)
#             y_pred = model(x_batch, attention_mask=(x_batch > 0).to(device)).to(device)
#             loss = F.binary_cross_entropy_with_logits(y_pred.view(-1), y_batch.float())
#             loss = loss.mean()
#             loss.backward()

#             if step % args.accumulation_steps == 0 or step == len(train_loader) - 1:
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 if not frozen:
#                     scheduler.step()
#                 else:
#                     scheduler0.step()
#             process_bar.set_postfix(loss=loss.item())
#             avg_loss += loss.item() / len(train_loader)
        
#         model.eval()
#         process_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), leave=False, desc="Validating...")
#         for step, (x_batch, y_batch) in process_bar:
#             x_batch = x_batch.to(device)
#             y_batch = y_batch.to(device)
#             with torch.no_grad():
#                 y_pred = model(x_batch, attention_mask=(x_batch > 0).to(device))
#                 y_pred = y_pred.squeeze().detach().cpu().numpy()
#                 val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
#         val_preds = sigmoid(val_preds)
#         best_th = 0
#         score = f1_score(y[val_idx], val_preds > 0.5)
#         print(f"\nAUC = {roc_auc_score(y[val_idx], val_preds):.4f}, F1 score @0.5 = {score:.4f}")
#         if score >= best_score:
#             torch.save(model.state_dict(),os.path.join(args.checkpoint_path, f"model_{fold}.bin"))
#             best_score = score
