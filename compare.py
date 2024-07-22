import csv
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np


TASK = 'power'

pmc_to_randomization1 = {}
if TASK in ['blinding', 'randomization', 'power']:
    for row in csv.reader(open('../covid_predictions_full_text.csv', 'r', encoding='iso-8859-1')):
        idx = {'randomization': -1, 'blinding': -2, 'power': -3}[TASK]
        pmcid = row[0].split('.')[0]
        if pmcid in pmc_to_randomization1:
            if not pmc_to_randomization1[pmcid]:
                pmc_to_randomization1[pmcid] = row[idx]
        else:
            pmc_to_randomization1[pmcid] = row[idx]
        if pmc_to_randomization1[pmcid] == 'FALSE':
            pmc_to_randomization1[pmcid] = False

pmc_to_randomization2 = {}
if TASK in ['randomization', 'blinding', 'ie']:
    idx = {'randomization': 0, 'blinding': 1, 'ie': -2}[TASK]
    for row in csv.reader(open('../full_text_rob_classified.csv', 'r')):
        pmc_to_randomization2[row[-1]] = float(row[idx]) > 0.5

pmc_to_randomization3 = {}
for f_name in os.listdir('../reports'):
    result = json.load(open(f'../reports/{f_name}/report.json', 'r'))
    idx = {'randomization': 'Randomization', 'blinding': 'Blinding', 'ie': 'Inclusion and Exclusion Criteria', 'power': 'Power Analysis'}[TASK]
    sent = [section['srList'][0]['sentence'] for section in result['rigor-table']['sections'] if section['title'] == idx][0]
    pmc_to_randomization3[f_name] = sent
    if pmc_to_randomization3[f_name] in ['not required.', 'not detected.']:
        pmc_to_randomization3[f_name] = False

doi_to_pmcid = {}
for row in csv.reader(open('../ids(1).csv', 'r')):
    doi_to_pmcid[row[2]] = row[1]
for row in csv.reader(open('../ids(2).csv', 'r')):
    doi_to_pmcid[row[2]] = row[1]

pmc_to_randomization4 = {}
flowyes_human = {}
if TASK == 'ie':
    for row in csv.reader(open('../barzooka_results.csv', 'r')):
        if row[-2] not in doi_to_pmcid:
            continue
        pmc_to_randomization4[doi_to_pmcid[row[-2]]] = bool(row[7])
        flowyes_human[doi_to_pmcid[row[-2]]] = bool(row[8])
pmc_to_true = {}
for row in csv.reader(open(TASK + '.csv', 'r')):
    row0 = row[0]
    if row[0] == '' and 'articles/' in row[1]:
        row0 = row[1].split('articles/')[1].split('/')[0]
    if row[2 if TASK != 'blinding' else 4] == {'ie': 'yes', 'power': '1', 'randomization': 'TRUE', 'blinding': 'Y'}[TASK]:
        pmc_to_true[row0] = True
    elif row[2 if TASK != 'blinding' else 4] == {'ie': 'no', 'power': '0', 'randomization': 'FALSE', 'blinding': 'N'}[TASK]:
        pmc_to_true[row0] = False
    else:
        print(row[2 if TASK != 'blinding' else 4])

pmcs = []
true = []
t1 = []
t2 = []
t3 = []
t4 = []
presumed_true = []
presumed_false = []
num_agreed_pos = 0
num_agreed_neg = 0
for pmc in pmc_to_randomization3:
    if pmc_to_randomization1:
        try:
            t1.append(pmc_to_randomization1[pmc] == 'TRUE')
        except:
            t1.append(False)
    if pmc_to_randomization2:
        try:
            t2.append(pmc_to_randomization2[pmc])
        except:
            t2.append(False)
    t3.append(bool(pmc_to_randomization3[pmc]))
    if pmc_to_randomization4:
        try:
            t4.append(pmc_to_randomization4[pmc])
        except:
            t4.append(False)
    s = 0
    t = 0
    if t1:
        t += 1
        s += int(t1[-1])
    if t2:
        t += 1
        s += int(t2[-1])
    if t3:
        t += 1
        s += int(t3[-1])
    if t4:
        t += 1
        s += int(t4[-1])
    if s / t == 0 or s / t == 1:
        true.append(t3[-1])
        if s / t == 0:
            num_agreed_neg += 1
        else:
            num_agreed_pos += 1
        if pmc in pmc_to_true:
            if s / t == 1:
                presumed_true.append(pmc_to_true[pmc])
            else:
                presumed_false.append(pmc_to_true[pmc])
    else:
        if pmc in pmc_to_true:
            if pmc in flowyes_human and flowyes_human[pmc]:
                true.append(True)
            else:
                true.append(pmc_to_true[pmc])
        elif pmc in flowyes_human and flowyes_human[pmc]:
            true.append(True)
        else:
            print('no true data', pmc, TASK)
            true.append(False)
    pmcs.append(pmc)

x = []
for i in range(len(t3)):
    x.append([])
    if t1:
        x[-1].append(t1[i])
    if t2:
        x[-1].append(t2[i])
    x[-1].append(t3[i])
    if t4:
        x[-1].append(t4[i])
        
model = LogisticRegression().fit(x, true)
ensemble = model.predict(x)
        




# from sklearn.model_selection import train_test_split
# res = []
# for i in range(1000):
#     x_train, x_test, y_train, y_test = train_test_split(x, true, test_size=0.5)
#     model = LogisticRegression().fit(x_train, y_train)
#     ensemble = model.predict(x)
#     s = ''
#     for a in [False, True]:
#         for b in [False, True]:
#             for c in [False, True]:
#                 s += str(model.predict([[a, b, c]])[0])
#     res.append(s)
# for s in list(set(res)):
#     print(res.count(s))




# try:
#     for a in [False, True]:
#         for b in [False, True]:
#             print(a, b, '|', model.predict([[a, b]])[0])
# except:
#     for a in [False, True]:
#         for b in [False, True]:
#             for c in [False, True]:
#                 print(a, b, c, '|', model.predict([[a, b, c]])[0])

t1 = np.array(t1)
t2 = np.array(t2)
t3 = np.array(t3)
t4 = np.array(t4)
ensemble = np.array(ensemble)
true = np.array(true)

# from irrCAC.raw import CAC
# import pandas as pd
# df = pd.DataFrame({'a': t1, 'b': true})
# print(CAC(df).gwet()['est']['coefficient_value'])

# df = pd.DataFrame({'a': t3, 'b': ensemble})
# print(CAC(df).gwet()['est']['coefficient_value'])

# df = pd.DataFrame({'a': t3, 'b': true})
# print(CAC(df).gwet()['est']['coefficient_value'])

# df = pd.DataFrame({'a': ensemble, 'b': true})
# print(CAC(df).gwet()['est']['coefficient_value'])

# t1 : Halil's tool
# t2 : Wang et al. RoB
# t3 : SciScore
# t4 : Barzooka

for i, t in enumerate([t1, t2, t3, t4, ensemble]):
    if len(t) == 0:
        continue

    tp = np.sum(t & true) * np.mean(presumed_true)
    fp = np.sum(t & ~true) * (1 - np.mean(presumed_false))
    fn = np.sum(~t & true) * np.mean(presumed_true)
    tn = np.sum(~t & ~true) * (1 - np.mean(presumed_false))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print(['Halil', 'Wang', 'SciScore', 'Barzooka', 'Ensemble'][i])
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('tp', tp)
    print('fp', fp)
    print('fn', fn)
    print('tn', tn)
    print()