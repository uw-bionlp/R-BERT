from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification




tokenizer = AutoTokenizer.from_pretrained('output/ctg_binary')
model = AutoModelForSequenceClassification.from_pretrained('output/ctg_binary')
ppln = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

data = []
with open('ctg_data/dev.tsv') as fin:
    for l in fin.readlines():
        ln = l.strip()
        if ln == '':
            continue
        data.append(ln.split('\t'))
        if len(data) == 500:
            break

correct = 0
predicted = ppln([ x[1] for x in data ])
for x, p in zip(data, predicted):
    text = x[1]
    label = x[-1]
    predicted = p['label']
    if (label == 'Other' and predicted == 'LABEL_0') or (label != 'Other' and predicted == 'LABEL_1'):
        correct += 1

score = correct / len(data)
print(score)