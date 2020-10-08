import os
import sys
import pandas as pd
from simpletransformers.classification import ClassificationModel


in_dir = 'ctg_data'
train = pd.read_csv(os.path.join(in_dir, 'train.tsv'), header=None, sep='\t', names=[ 'id', 'text', 'target', 'label'])
dev = pd.read_csv(os.path.join(in_dir, 'dev.tsv'), header=None, sep='\t', names=[ 'id', 'text', 'target', 'label'])

train.loc[train['label'] == 'Other', 'target'] = 0
train.loc[train['label'] != 'Other', 'target'] = 1
dev.loc[dev['label'] == 'Other', 'target'] = 0
dev.loc[dev['label'] != 'Other', 'target'] = 1

del train['id']
del train['label']
del dev['id']
del dev['label']

print(train.head(5))
print(dev.head(5))

additional_special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]
with open(os.path.join('ctg_data', 'types.tsv')) as fin:
    types = [ t.strip() for t in fin.readlines() ]
    for t in types:
        additional_special_tokens.append(t)

train_args = {
    'evaluate_during_training': True,
    'logging_steps': 1000,
    'num_train_epochs': 5,
    'evaluate_during_training_steps': 1000,
    'save_eval_checkpoints': False,
    'train_batch_size': 32,
    'eval_batch_size': 64,
    'overwrite_output_dir': True,
    'output_dir': 'output/ctg_binary',
    'fp16': False,
    'special_tokens': additional_special_tokens,
    'cache_dir': in_dir
}
model = ClassificationModel('bert', 'allenai/scibert_scivocab_uncased', num_labels=2, use_cuda=False, cuda_device=None, args=train_args)
model.train_model(train, eval_df=dev)