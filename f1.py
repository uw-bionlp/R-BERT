import os
import sys
import numpy as np
from sklearn.metrics import f1_score

predict_file = sys.argv[1]
gold_file = sys.argv[2]

def main():
    with open(predict_file) as fin:
        predict = []
        for l in fin.readlines():
            if l.strip() != '':
                predict.append(l.split('\t')[1].strip())
    with open(gold_file) as fin:
        gold = [] 
        for l in  fin.readlines():
            if l.strip() != '':
                parts = l.split('\t')
                gold.append({ 'text': parts[1], 'label': parts[3].strip() })

    assert len(predict) == len(gold), f'Predict'

    gold_labels = [ g['label'] for g in gold ]
    acc = (np.array(predict) == np.array(gold_labels)).mean()
    # f1 = f1_score(y_true=gold_labels, y_pred=predict, average='micro')

    print(f'Overall - {acc} (n={len(gold_labels)})')

    unq_labels = list(set(gold_labels))

    for label in sorted(unq_labels):
        label_g = [ (i,g) for i,g in enumerate(gold_labels) if g == label ]
        idxs = [ l[0] for l in label_g ] 
        label_p = [ (i,p) for i,p in enumerate(predict) if i in idxs ]

        acc = (np.array(label_g) == np.array(label_p)).mean()
        print(f'    {label} - {acc} (n={len(label_g)})')

    print('')
    i = 0
    for g, p in zip(gold, predict):
        if g['label'] != p:
            print(f'"{g["text"]}"')
            print(f'    Pred: {p}')
            print(f'    True: {g["label"]}\n')
            i += 1

    x=1

    '''
    tp, tn, fp, fn = 0, 0, 0, 0
    type_scores = {}

    for p, g in zip(predict, gold):
        if not type_scores.get(p): type_scores[p] = { 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0 }
        if not type_scores.get(g['label']): type_scores[g['label']] = { 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0 }
        if g['label'] == p:
            tp += 1
            type_scores[p]['tp'] += 1
        else:
            fp += 1
            fn += 1
            type_scores[p]['fp'] += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f'Overall')
    print(f'    Precision: {round(precision*100,1)}')
    print(f'    Recall: {round(recall*100,1)}')
    print(f'    F1: {round(f1*100,1)}')
    print('')

    for k,v in sorted(type_scores.items()):
        precision = v['tp'] / (v['tp'] + v['fp']) if v['tp'] + v['fp'] > 0 else 0.0
        recall = v['tp'] / (v['tp'] + v['fn']) if v['tp'] + v['fn'] > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        print(f'    {k}')
        print(f'        Precision: {round(precision*100,1)}')
        print(f'        Recall: {round(recall*100,1)}')
        print(f'        F1: {round(f1*100,1)}')
    '''

main()