import os
import sys

in_dir = 'ctg_data_all'
pred_file = os.path.join('eval', 'ctg_all.txt')


def clean_label(label):
    if ':' in label:
        cleaned = label[label.find(':')+1:]
        cleaned = cleaned[:cleaned.find('(')]
        return cleaned
    return label


def main():
    with open(os.path.join(in_dir, 'relation_combinations.tsv')) as fin:
        known_rels = { tuple(l.strip().split('\t')) for l in fin.readlines() if l != '' }

    with open(os.path.join(in_dir, 'dev.tsv')) as fin:
        gold = [ l.strip().split('\t') for l in fin.readlines() if l != '' ]

    with open(pred_file) as fin:
        pred = [ l.strip().split('\t')[1] for l in fin.readlines() if l != '' ]

    with open(os.path.join(in_dir, 'relation2id.tsv')) as fin:
        relation2id = { r.strip():i for i, r in enumerate(fin.readlines()) if r != '' }

    fixable, i = 0, 0
    for p, g in zip(pred, gold):
        id, text, label_id, label = g
        orig_label = label
        orig_p = p
        label = clean_label(label)
        p = clean_label(p)

        e11_idx = text.find('[E11] ')
        e21_idx = text.find('[E21] ')
        bound_len = len('[E11]')+1
        
        e1_end = text[e11_idx+bound_len:].find(' ')
        e2_end = text[e21_idx+bound_len:].find(' ')

        e1 = text[e11_idx+bound_len:][:e1_end]
        e2 = text[e21_idx+bound_len:][:e2_end]

        #if p != label:
        #    print(text)
        #    print(f'Predicted: {orig_p}')
        #    print(f'   Actual: {orig_label}\n')

        pred_rel = (e1, e2, p) if 'E1,E2' in orig_p else (e2, e1, p)
        possible = pred_rel in known_rels
        
        if p != 'Other' and not possible:
            fixable += 1
            print("\t".join([ str(i), 'Other' ]))
        else:
            print("\t".join([ str(i), orig_p ]))
        i += 1

    #print(fixable)

main()
