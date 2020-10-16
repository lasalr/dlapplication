import sys

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
import ipywidgets as widgets
import os
import math

plt.rcParams["figure.figsize"] = (16,8)


def checkPrediction(prediction, label):
    correctness = 0
    if len(label) == 1 and len(prediction) == 1:
        correctness = int(abs(label[0] - prediction[0]) < 0.1)
    elif len(label) == 1 and len(prediction) != 1:
        correctness = int(label[0] == np.argmax(prediction))
    elif len(label) != 1 and len(prediction) != 1:
        correctness = int(np.argmax(label) == np.argmax(prediction))
    return correctness

# dirs = [d for d in sorted(os.listdir('.')) if os.path.isdir(d)]
# # wFolder = widgets.Dropdown(
# #     options=dirs,
# #     description='Experiment:',
# # )
# # display(wFolder)

experimentFolder = './examples/averagingAtTheEnd/Results/Results-HPC/LinearSVC_Averaging_2020-10-12_23-19-27'

# experimentFolder = wFolder.value
print(experimentFolder)

nodesAmount = 0
dirs = [d for d in os.listdir(experimentFolder) if os.path.isdir(os.path.join(experimentFolder,d))]
for d in dirs:
    if 'worker' in d:
        nodesAmount += 1
print("Learners amount is ", str(nodesAmount))

displayStep = 100
# should be larger or equal to displayStep or it will just record same image several times
recordStep = 100
recordUnique = False


files = []
correctSums = []
accuracies = []
for i in range(nodesAmount):
    files.append(open(os.path.join(experimentFolder, "worker" + str(i), "predictions.txt"), "r"))
    accuracies.append([0])
    correctSums.append([0])
t = [0]
commonStep = 0

average_acc = 0

for j in range(2):
    for i in range(nodesAmount):
        file = files[i]
        where = file.tell()
        line = file.readline()
        if not line:
            time.sleep(1)
            file.seek(where)
        else:
            pred = [float(x) for x in line[:-1].split('\t')[1].split(',')]
            label = [float(x) for x in line[:-1].split('\t')[2].split(',')]
            correctness = checkPrediction(pred, label)
            correctSums[i].append(correctSums[i][-1] + correctness)
            accuracies[i].append(correctSums[i][-1]*100.0/(commonStep+1))
    currentStep = min([len(a) for a in accuracies])
    if currentStep > commonStep:
        commonStep = currentStep
        cutAccuracies = [a[1:commonStep] for a in accuracies]
        mu = np.array(cutAccuracies).mean(axis=0)
        sigma = np.array(cutAccuracies).std(axis=0)
        average_acc = mu
        t.append(t[-1] + 1)

print('Average accuracy={}'.format(average_acc))
print(type(average_acc))