import re
import json
from sklearn.linear_model import LogisticRegression


pcl_fp = ['PMC8862453', 'PMC8344454', 'PMC8256281', 'PMC8322202', 'PMC8862453', 'PMC8458311', 'PMC8869555', 'PMC8983315', 'PMC8545493', 'PMC8977537']
file = open('pcl-Compare.html', 'r').read()
count = 0
pmcid_to_pcl = {}
outs = []
for pmcid in list(set(re.findall('PMC[0-9]+', file))):
    pmcid_to_pcl[pmcid] = f'target="blank">5-Compare/{pmcid}.txt</a></td></tr></table>' not in file
pmcid_to_sciscore = {}
for line in open('/Users/petereckmann/Downloads/Downloads/cell_line_out.csv', 'r'):
    pmcid, _, label = line.strip().split(',')
    pmcid_to_sciscore[pmcid] = label.lower() == 'y'
sciscore = []
pcl = []
t = []
for pmcid in pmcid_to_pcl:
    if pmcid not in pmcid_to_sciscore:
        pmcid_to_sciscore[pmcid] = False
    if pmcid_to_pcl[pmcid] == pmcid_to_sciscore[pmcid]:
        # print(f'{pmcid},{pmcid_to_pcl[pmcid]},{pmcid_to_sciscore[pmcid]},{pmcid_to_sciscore[pmcid]}')

        t.append(pmcid_to_sciscore[pmcid])
    elif pmcid in pcl_fp:
        # print(f'{pmcid},{pmcid_to_pcl[pmcid]},{pmcid_to_sciscore[pmcid]},False')
        t.append(False)
    else:
        # print(f'{pmcid},{pmcid_to_pcl[pmcid]},{pmcid_to_sciscore[pmcid]},True')
        t.append(True)
    pcl.append(pmcid_to_pcl[pmcid])
    sciscore.append(pmcid_to_sciscore[pmcid])


model = LogisticRegression().fit(list(zip(pcl, sciscore)), t)
ensemble = model.predict(list(zip(pcl, sciscore)))

x = list(zip(pcl, sciscore))
from sklearn.model_selection import train_test_split
res = []
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, t, test_size=0.2)
    model = LogisticRegression().fit(x_train, y_train)
    ensemble = model.predict(x)
    s = ''
    for a in [False, True]:
        for b in [False, True]:
            s += str(model.predict([[a, b]])[0])
    res.append(s)
for s in list(set(res)):
    print('percent same', res.count(s) / 1000)


import numpy as np
pcl = np.array(pcl, dtype=bool)
sciscore = np.array(sciscore, dtype=bool)
ensemble = np.array(ensemble, dtype=bool)
print(ensemble[~pcl & sciscore])

from irrCAC.raw import CAC
import pandas as pd

print('-----')
df = pd.DataFrame({'a': pcl, 'b': sciscore})
print(CAC(df).gwet()['est']['coefficient_value'])

print('-----')

df = pd.DataFrame({'a': pcl, 'b': ensemble})
print(CAC(df).gwet()['est']['coefficient_value'])

print('-----')
df = pd.DataFrame({'a': pcl, 'b': t})
print(CAC(df).gwet()['est']['coefficient_value'])

print('-----')
df = pd.DataFrame({'a': sciscore, 'b': ensemble})
print(CAC(df).gwet()['est']['coefficient_value'])

print('-----')
df = pd.DataFrame({'a': sciscore, 'b': t})
print(CAC(df).gwet()['est']['coefficient_value'])

print('-----')
df = pd.DataFrame({'a': ensemble, 'b': t})
print(CAC(df).gwet()['est']['coefficient_value'])

model = LogisticRegression().fit(x, t)
ensemble = model.predict(x)

for a in [False, True]:
    for b in [False, True]:
        print(a, b, '|', model.predict([[a, b]])[0])
exit()

import sklearn.metrics
print(sklearn.metrics.precision_score(t, pcl))
print(sklearn.metrics.precision_score(t, sciscore))
print(sklearn.metrics.precision_score(t, ensemble))
print()
print(sklearn.metrics.recall_score(t, pcl))
print(sklearn.metrics.recall_score(t, sciscore))
print(sklearn.metrics.recall_score(t, ensemble))
print()
print(sklearn.metrics.accuracy_score(t, pcl))
print(sklearn.metrics.accuracy_score(t, sciscore))
print(sklearn.metrics.accuracy_score(t, ensemble))
print()
print(sklearn.metrics.f1_score(t, pcl))
print(sklearn.metrics.f1_score(t, sciscore))
print(sklearn.metrics.f1_score(t, ensemble))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
tn, fp, fn, tp = confusion_matrix(t, ensemble).ravel()
print('tp', tp)
print('fp', fp)
print('fn', fn)
print('tn', tn)
exit()