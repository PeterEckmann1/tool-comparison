import csv
import os
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import sqlite3


TASK = 'code'

pmc_to_tool1 = {}
if TASK in ['blinding', 'randomization', 'power']:
    for row in csv.reader(open('../covid_predictions_full_text.csv', 'r', encoding='iso-8859-1')):
        idx = {'randomization': -1, 'blinding': -2, 'power': -3}[TASK]
        pmcid = row[0].split('.')[0]
        if pmcid in pmc_to_tool1:
            if not pmc_to_tool1[pmcid]:
                pmc_to_tool1[pmcid] = row[idx]
        else:
            pmc_to_tool1[pmcid] = row[idx]
        if pmc_to_tool1[pmcid] == 'FALSE':
            pmc_to_tool1[pmcid] = False

pmc_to_tool2 = {}
if TASK in ['randomization', 'blinding', 'ie']:
    idx = {'randomization': 0, 'blinding': 1, 'ie': -2}[TASK]
    for row in csv.reader(open('../full_text_rob_classified.csv', 'r')):
        pmc_to_tool2[row[-1]] = float(row[idx]) > 0.5

pmc_to_tool3 = {}
for f_name in os.listdir('../reports'):
    result = json.load(open(f'../reports/{f_name}/report.json', 'r'))
    idx = {'randomization': 'Randomization', 'blinding': 'Blinding', 'ie': 'Inclusion and Exclusion Criteria', 'power': 'Power Analysis', 'code': 'Code Information', 'data': 'Data Information'}[TASK]
    if TASK in ['code', 'data']:
        try:
            sent = [section['srList'][0]['sentence'] for section in result['rigor-table']['sections'] if section['title'] == idx][0]
        except:
            sent = 'not detected.'
    else:
        sent = [section['srList'][0]['sentence'] for section in result['rigor-table']['sections'] if section['title'] == idx][0]
    pmc_to_tool3[f_name] = sent
    if pmc_to_tool3[f_name] in ['not required.', 'not detected.']:
        pmc_to_tool3[f_name] = False

doi_to_pmcid = {}
for row in csv.reader(open('../ids(1).csv', 'r')):
    doi_to_pmcid[row[2]] = row[1]
for row in csv.reader(open('../ids(2).csv', 'r')):
    doi_to_pmcid[row[2]] = row[1]

pmc_to_tool4 = {}
flowyes_human = {}
if TASK == 'ie':
    for row in csv.reader(open('../barzooka_results.csv', 'r')):
        if row[-2] not in doi_to_pmcid:
            continue
        pmc_to_tool4[doi_to_pmcid[row[-2]]] = bool(row[7])
        flowyes_human[doi_to_pmcid[row[-2]]] = bool(row[8])

pmc_to_tool5 = {}
if TASK in ['code', 'data']:
    conn = sqlite3.connect('../data.db')
    cur = conn.cursor()
    for pmcid, oddpub_full_open_code, oddpub_full_open_data, sciscore_methods_open_code, sciscore_methods_open_data in cur.execute('select pmcid, oddpub_full_open_code, oddpub_full_open_data, sciscore_methods_open_code, sciscore_methods_open_data from papers'):
        pmc_to_tool5[pmcid] = bool(oddpub_full_open_code if TASK == 'code' else oddpub_full_open_data)
        pmc_to_tool3[pmcid] = bool(sciscore_methods_open_code if TASK == 'code' else sciscore_methods_open_data)

pmc_to_true = {}
for row in csv.reader(open(TASK + '.csv', 'r')):
    row0 = row[0]
    if row[0] == '' and 'articles/' in row[1]:
        row0 = row[1].split('articles/')[1].split('/')[0]
    if row[2 if TASK != 'blinding' else 4] == {'ie': 'yes', 'power': '1', 'randomization': 'TRUE', 'blinding': 'Y', 'code': '1', 'data': '1'}[TASK]:
        pmc_to_true[row0] = True
    elif row[2 if TASK != 'blinding' else 4] == {'ie': 'no', 'power': '0', 'randomization': 'FALSE', 'blinding': 'N', 'code': '0', 'data': '0'}[TASK]:
        pmc_to_true[row0] = False

pmcs = []
true = []
t1 = []
t2 = []
t3 = []
t4 = []
t5 = []
presumed_true = []
presumed_false = []
num_agreed_pos = 0
num_agreed_neg = 0

num_used = 0

for pmc in pmc_to_tool3:
    if pmc_to_tool1:
        try:
            t1.append(pmc_to_tool1[pmc] == 'TRUE')
        except:
            t1.append(False)
    if pmc_to_tool2:
        try:
            t2.append(pmc_to_tool2[pmc])
        except:
            t2.append(False)
    t3.append(bool(pmc_to_tool3[pmc]))
    if pmc_to_tool4:
        try:
            t4.append(pmc_to_tool4[pmc])
        except:
            t4.append(False)
    if pmc_to_tool5:
        try:
            t5.append(pmc_to_tool5[pmc])
        except:
            t5.append(False)

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
    if t5:
        t += 1
        s += int(t5[-1])
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
        num_used += 1
        if pmc in pmc_to_true:
            if TASK == 'ie' and pmc in flowyes_human and flowyes_human[pmc]:
                true.append(True)
            else:
                true.append(pmc_to_true[pmc])
        elif TASK == 'ie' and pmc in flowyes_human and flowyes_human[pmc]:
            true.append(True)
        else:
            print('no true data', pmc, TASK)
            # print(f'{pmc},https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/,')
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
    if t5:
        x[-1].append(t5[i])
        
model = LogisticRegression().fit(x, true)
ensemble = model.predict(x)
        




# from sklearn.model_selection import train_test_split
# res = []
# for i in range(1000):
#     x_train, x_test, y_train, y_test = train_test_split(x, true, test_size=0.2)
#     model = LogisticRegression().fit(x_train, y_train)
#     ensemble = model.predict(x)
#     s = ''
#     for a in [False, True]:
#         for b in [False, True]:
#             s += str(model.predict([[a, b]])[0])
#     res.append(s)
# for s in list(set(res)):
#     print('percent same', res.count(s) / 1000)



t1 = np.array(t1)
t2 = np.array(t2)
t3 = np.array(t3)
t4 = np.array(t4)
t5 = np.array(t5)
ensemble = np.array(ensemble)
true = np.array(true)

if len(presumed_true) + len(presumed_false) > 100:
    if len(presumed_false) < 50:
        presumed_true = presumed_true[-(100 - len(presumed_false)):]
    elif len(presumed_true) < 50:
        presumed_false = presumed_false[-(100 - len(presumed_true)):]
    else:
        presumed_true = presumed_true[-50:]
        presumed_false = presumed_false[-50:]

tool_names = []
nums = []
for i, t in enumerate([t1, t2, t3, t4, t5, ensemble]):
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

    # print(['Halil', 'Wang', 'SciScore', 'Barzooka', 'Ensemble'][i])
    # print('Accuracy:', accuracy)
    # print('Precision:', precision)
    # print('Recall:', recall)
    # print('F1:', f1)
    # print('tp', tp)
    # print('fp', fp)
    # print('fn', fn)
    # print('tn', tn)
    # print()

    # print('''
    # \\begin{table*}
    # \\caption{\\textbf{Confusion matrix _A_ _B_}. a.}
    # \\begin{center}
    # \\begin{small}
    # \\begin{sc}
    # \\begin{tabular}{cccccc}
    # \\toprule
    # & Predicted positive & Predicted negative\\\\
    # \midrule
    # Actual positive & _TP_ & _FN_\\\\
    # Actual negative & _FP_ & _TN_\\\\
    # \\bottomrule
    # \end{tabular}
    # \end{sc}
    # \end{small}
    # \end{center}
    # \end{table*}'''.replace('_A_', {'random': 'randomization', 'blind': 'blinding', 'ie': 'inclusion and exclusion criteria', 'power': 'power analysis'}[TASK])
    #     .replace('_B_', ['CONSORT-TM', 'pre-rob', 'SciScore', 'Barzooka', 'Ensemble'][i])
    #     .replace('_TP_', str(round(tp, 1)))
    #     .replace('_FP_', str(round(fp, 1)))
    #     .replace('_FN_', str(round(fn, 1)))
    #     .replace('_TN_', str(round(tn, 1))))
    # print()

    tool_names.append(['CONSORT-TM', 'pre-rob', 'SciScore', 'Barzooka', 'ODDPub', 'Ensemble'][i])
    nums.append([round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)])



# try:
#     for a in [False, True]:
#         for b in [False, True]:
#             print(a, b, '|', model.predict([[a, b]])[0])
# except:
#     for a in [False, True]:
#         for b in [False, True]:
#             for c in [False, True]:
#                 print(a, b, c, '|', model.predict([[a, b, c]])[0])
# print(tool_names)
# exit()


latex = '''\\begin{table*}
\\caption{\\textbf{Estimated tool performance statistics for _A_ detection}. Statistics are estimated based on the gold standard test set of 1,500 papers (see Methods for details). The Ensemble tool uses a combination of the results from all tools to make a classification.}
\\label{tab:performance-_D_}
\\begin{center}
\\begin{small}
\\begin{sc}
\\begin{tabular}{c|cccc}
\\toprule
Tool & Accuracy & Precision & Recall & F1\\\\
\midrule
'''

for i in range(len(tool_names)):
    row = [(('\\textbf{' + str(nums[i][j]) + '}') if nums[i][j] == np.max(nums, axis=0)[j] else str(nums[i][j])) for j in range(len(nums[i]))]
    if tool_names[i] == 'Ensemble':
        latex += '\\midrule \n'
    latex += tool_names[i] + ' & ' + ' & '.join(row) + ' \\\\ \n'

latex += '''\\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\end{table*}'''

latex = latex.replace('_A_', {'randomization': 'randomization', 'blinding': 'blinding', 'ie': 'inclusion and exclusion criteria', 'power': 'power analysis', 'code': 'open code', 'data': 'open data'}[TASK])
latex = latex.replace('_D_', TASK)

print(latex)


latex = '''\\begin{table*}
\\caption{\\textbf{Gwet's agreement among each tool and the gold standard (``true'') for _C_}.}
\\label{tab:agreement-_C_}
\\begin{center}
\\begin{small}
\\begin{sc}
\\begin{tabular}{_A_}
\\toprule
_B_\\
\\midrule
'''

latex = latex.replace('_C_', {'randomization': 'randomization', 'blinding': 'blinding', 'ie': 'inclusion and exclusion criteria', 'power': 'power analysis', 'code': 'open code', 'data': 'open data'}[TASK])

tool_names += ['True']

latex = latex.replace('_A_', 'c' * (len(tool_names) + 2))
latex = latex.replace('_B_', ' & ' + ' & '.join(tool_names) + '\\')
latex = latex.replace('_C_', TASK)

from irrCAC.raw import CAC
import pandas as pd
res = [a for a in [t1, t2, t3, t4, t5, ensemble, true] if len(a) > 0]
for i in range(len(tool_names)):
    vals = []
    for j in range(len(tool_names)):
        df = pd.DataFrame({'a': res[i], 'b': res[j]})
        gwet = float(CAC(df).gwet()['est']['coefficient_value'])
        if i <= j:
            vals.append(str(round(gwet, 2)))
        else:
            vals.append(' ')
    latex += tool_names[i] + ' & ' + ' & '.join(vals) + '\\\\\n'

latex += '''\\bottomrule
\end{tabular}
\end{sc}
\end{small}
\end{center}
\end{table*}'''

print('\n\n\n\n\n')
print(latex)