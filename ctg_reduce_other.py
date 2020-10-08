import os
import sys
import random

in_dir = 'ctg_data'
out_dir = 'ctg_data_reduced_other'
out_path = os.path.join(out_dir, 'train.tsv')

others, no_others = [], []
for fl in [ 'train.tsv', 'test.tsv' ]:
    with open(os.path.join(in_dir, fl)) as fin:
        for l in fin.readlines():
            ln = l.strip()
            if ln == '':
                continue
            label = ln.split('\t')[-1]
            if 'Name(' in label:
                continue
            if label == 'Other':
                others.append(ln)
            else:
                no_others.append(ln)
        
others_cnt = len(others)        
no_others_cnt = len(no_others)
desired_others_cnt = int(no_others_cnt)

sampled = random.sample(others, desired_others_cnt)
sample_len = len(sampled)

reordered = no_others + sampled
reordered = random.sample(reordered, len(reordered))
i = 1
with open(out_path, 'w+') as fout:
    for r in reordered:
        parts = r.split('\t')
        parts[0] = str(i)
        fout.write('\t'.join(parts) + '\n')
        i += 1