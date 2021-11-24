import csv, math
from polyleven import levenshtein
from tqdm import trange

DATASET = '../../Dataset/Text-Data/RP-Mod-Crowd.csv'
NAME = DATASET.split('/')[-1]
TRESHOLD_NEAR = 0.15

rows_list = []
with open(DATASET) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows_list.append(row)
        
real_duplicates = []
near_duplibates = []
for _ in trange(len(rows_list)):
    base_row = rows_list.pop()
    for row in rows_list:
        length = max(len(base_row['Text']), len(row['Text']))
        max_ths = math.ceil(length * TRESHOLD_NEAR)
        dist = levenshtein(base_row['Text'], row['Text'], max_ths)
        if dist == 0 and base_row['id'] not in {r['id'] for r in real_duplicates}:
            real_duplicates.append(base_row['Text'])
        if 1 <= dist <= max_ths and base_row['id'] not in {r['id'] for r in near_duplibates}:
            near_duplibates.append(base_row['Text'])
            
print(f'Real duplicates: {len(real_duplicates)}')
print(f'Near duplicates: {len(near_duplibates)}')

with open('real_duplicates.csv', 'w')  as f:
    dict_writer = csv.DictWriter(f, real_duplicates[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(real_duplicates)
    
with open('near_duplibates.csv', 'w')  as f:
    dict_writer = csv.DictWriter(f, near_duplibates[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(near_duplibates)