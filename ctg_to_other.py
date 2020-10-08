import os
import sys

in_dir = 'ctg_data'
out_dir = 'ctg_data_other'

dev = 'dev.tsv'
test = 'test.tsv'
train = 'train.tsv'
rel = 'relation2id.tsv'

rels = [ 'Other', 'Not-Other' ]
relation2id, id2relation = {}, []
with open(os.path.join(out_dir, rel), 'w+') as fout:
    for rel in rels:
        if rel not in relation2id:
            id2relation.append(rel)
            relation2id[rel] = len(relation2id)
            fout.write(f'{rel}\n')

for f in [ train, test, dev ]:
    in_path = os.path.join(in_dir, f)
    out_path = os.path.join(out_dir, f)
    data, i = [], 1
    with open(in_path) as fin:
        with open(out_path, 'w+') as fout:
            for l in fin.readlines():
                ln = l.strip()
                if ln == '':
                    continue
                parts = ln.split('\t')
                label = parts[-1]
                if label != 'Other':
                    label = 'Not-Other'
                    parts[-1] = 'Not-Other'
                parts[0] = str(i)
                parts[-2] = str(relation2id[label])
                fout.write('\t'.join(parts) + '\n')
                i += 1