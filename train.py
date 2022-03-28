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
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from transformers.modeling_utils import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from utils import *
from models import *
import logging
from tqdm import tqdm
tqdm.pandas()


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
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


logging.info("Creating RDRSegmenter...")
args = parser.parse_args()
bpe = fastBPE(args)
rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')

seed_everything(args.seed)
if torch.cuda.is_available():
    logging.info('Using GPU')
    device = torch.device('cuda')
else:
    logging.info('Using CPU')
    device = torch.device('cpu')


logging.info('Loading dictionary...')
# Load the Dictionary
vocab = Dictionary()
vocab.add_from_file(args.dict_path)

# Load data
logging.info("Loading data...")
train_df = pd.read_csv(args.train_path)

# Tokenize
logging.info("Tokenizing...")
train_df["text"] = train_df["text"].progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
X_train = convert_lines(train_df, vocab, bpe, args.max_sequence_length)

# Label encoder
lb = LabelEncoder()
y = train_df.label.values
y = lb.fit_transform(y)

# Load model
logging.info("Loading model...")
config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_labels=1
)

model = RobertaVN.from_pretrained(args.pretrained_path, config=config)
# model.eval()
model.to(device)

if torch.cuda.device_count():
    logging.info(f"Training using {torch.cuda.device_count()} gpus")
    model = nn.DataParallel(model)
    tsfm = model.module.roberta
else:
    tsfm = model.roberta


# Optimizer and lr schedulers
logging.info("Starting training...")
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(args.epochs * len(train_df) / args.batch_size / args.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
# Warmup 100 step before start learning
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)
scheduler0 = get_constant_schedule(optimizer)

# Checkpoint path 
if not os.path.exists(args.checkpoint_path):
    os.mkdir(args.checkpoint_path)


X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, stratify=y)

best_score = 0
early_stopping = 0
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long))
valid_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.long))
for child in tsfm.children():
    for param in child.parameters():
        if not param.requires_grad:
            pass
        param.requires_grad = False

frozen = True
for epoch in range(args.epochs):
    if epoch > 0 and frozen:
        for child in tsfm.children():
            for param in child.parameters():
                param.requires_grad = True
        frozen = False
        del scheduler0
        torch.cuda.empty_cache()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    total_loss = 0.
    train_preds = None
    
    optimizer.zero_grad()
    model.train()   
    with tqdm(enumerate(train_loader), total=len(train_loader), unit="batch") as process_bar:
        for step, (x_batch, y_batch) in process_bar:
            process_bar.set_description(f"Epoch {epoch}...")
            
            y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device))
            predictions = y_pred.squeeze().detach().cpu().numpy()
            train_preds = np.atleast_1d(predictions) if train_preds is None else np.concatenate([train_preds, np.atleast_1d(predictions)])
            # compute loss
            loss = F.binary_cross_entropy_with_logits(y_pred.view(-1).to(device), y_batch.float().to(device))
            loss = loss.mean()
            loss.backward()

            if step % args.accumulation_steps == 0 or step == len(train_loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()
            process_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()
    train_preds = sigmoid(train_preds)
    train_acc = accuracy_score(y_train, train_preds > 0.5) * 100
    train_f1 = f1_score(y_train, train_preds > 0.5)
    logging.info("Accuracy: {0:.4f}%".format(train_acc))
    logging.info("F1 Score: {0:.4f}".format(train_f1))
    logging.info("Average training loss: {0:.4f}".format(total_loss))

    model.eval()
    val_preds = None
    process_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc="Validating...")
    for step, (x_batch, y_batch) in process_bar:
        with torch.no_grad():
            y_pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device))
            y_pred = y_pred.squeeze().detach().cpu().numpy()
            # convert to 1d array
            val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
    val_preds = sigmoid(val_preds)

    score = f1_score(y_val, val_preds > 0.5)
    val_accuracy = accuracy_score(y_val, val_preds > 0.5)
    logging.info(f"\nAUC: {roc_auc_score(y_val, val_preds):.4f}, F1 score @0.5: {score:.4f}, Accuracy: {round(val_accuracy * 100, 2)}%")
    if score >= best_score:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"model_{epoch}.bin"))
        best_score = score
    else:
        early_stopping += 1
    
    if early_stopping > 10:
        logging.info("Early stopping at epoch {}".format(epoch))
        break