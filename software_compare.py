import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def normalize(str):
    return str.replace('-', '-').replace('‐', '-')


sciscore_hashes = []
for row in csv.reader(open('../sciscore_v3_software_tools.csv', 'r')):
    if row[0] != 'PMCID':
        sciscore_hashes.append(row[0] + normalize(row[1]))
softcite_hashes = []
for row in csv.reader(open('../mentions_one_per_row.csv', 'r')):
    if row[0] != 'article_pmcid' and row[1].strip() != '' and row[1] in row[2]:
        softcite_hashes.append(row[0] + normalize(row[1]))
full_list = list(set(sciscore_hashes + softcite_hashes))
hash_to_decision = {}
for row in csv.reader(open('software.csv', 'r')):
    if row[3] == 'Yes':
        decision = True
    elif row[3] == 'No':
        decision = False
    else:
        print(row[0], row[1], row[3])
    hash_to_decision[row[0] + normalize(row[1])] = decision
for row in csv.reader(open('../software_disagreements.csv', 'r')):
    hash_to_decision[row[0] + normalize(row[1])] = row[3] == 'Yes'


real = []
sciscore = []
softcite = []
agreed = []
for h in full_list:
    if h in softcite_hashes and h in sciscore_hashes:
        agreed.append(h)
        real.append(True)
    else:
        try:
            real.append(hash_to_decision[h])
        except:
            real.append(True) #manually looked at the disagreements and they were all correct
    sciscore.append(h in sciscore_hashes)
    softcite.append(h in softcite_hashes)
real = np.array(real)
sciscore = np.array(sciscore)
softcite = np.array(softcite)


# import random
# random.shuffle(agreed)
# print('\n'.join(agreed[:100]))


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


from sklearn.linear_model import LogisticRegression
x = list(zip(sciscore, softcite))
model = LogisticRegression().fit(x, real)
ensemble = model.predict(x)

for a in [False, True]:
    for b in [False, True]:
        print(a, b, '|', model.predict([[a, b]])[0])

# from sklearn.model_selection import train_test_split
# res = []
# for i in range(1000):
#     x_train, x_test, y_train, y_test = train_test_split(x, real, test_size=0.2)
#     model = LogisticRegression().fit(x_train, y_train)
#     ensemble = model.predict(x)
#     s = ''
#     for a in [False, True]:
#         for b in [False, True]:
#             s += str(model.predict([[a, b]])[0])
#     res.append(s)
# for s in list(set(res)):
#     print('percent same', res.count(s) / 1000)
# exit()

# 100% of randomly selected software was correct

# print(accuracy_score(real, ensemble))
# print(precision_score(real, ensemble))
# print(recall_score(real, ensemble))
# print(f1_score(real, ensemble))

# exit()

# tn, fp, fn, tp = confusion_matrix(real, ensemble).ravel()
# print('tp', tp)
# print('fp', fp)
# print('fn', fn)
# print('tn', tn)
# exit()

from irrCAC.raw import CAC
import pandas as pd

print('-----')
df = pd.DataFrame({'a': sciscore, 'b': softcite})
print(CAC(df).gwet()['est']['coefficient_value'])

df = pd.DataFrame({'a': sciscore, 'b': ensemble})
print(CAC(df).gwet()['est']['coefficient_value'])

df = pd.DataFrame({'a': sciscore, 'b': real})
print(CAC(df).gwet()['est']['coefficient_value'])

df = pd.DataFrame({'a': ensemble, 'b': softcite})
print(CAC(df).gwet()['est']['coefficient_value'])

df = pd.DataFrame({'a': real, 'b': softcite})
print(CAC(df).gwet()['est']['coefficient_value'])

df = pd.DataFrame({'a': ensemble, 'b': real})
print(CAC(df).gwet()['est']['coefficient_value'])