import csv, random, sys
sys.path.append('preprocessing')
from cleaning import clean_text

data = []
with open('data/emails_raw.csv', mode='r', encoding='utf-8', errors='ignore') as f:
    reader = csv.DictReader(f)
    for row in reader:
        t = row.get('text', '')
        lbl = row.get('label', '0')
        cleaned = clean_text(t)
        if len(cleaned) > 0:
            data.append({'text': cleaned, 'raw_text': t, 'label': lbl})

random.seed(42)
random.shuffle(data)

n = len(data)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = data[:train_end]
val = data[train_end:val_end]
test = data[val_end:]

def write_csv(filename, rows):
    with open(filename, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'raw_text', 'label'])
        writer.writeheader()
        writer.writerows(rows)

write_csv('data/train.csv', train)
write_csv('data/val.csv', val)
write_csv('data/test.csv', test)

print(f'train: {len(train)}')
print(f'val: {len(val)}')
print(f'test: {len(test)}')
