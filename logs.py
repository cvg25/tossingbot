import numpy as np

path = './training_data/1730844909/experiences.csv'

success_grasps = []
success_throws = []
explore_grasps = []
explore_throws = []

with open(path, 'r') as file:
    for idx, line in enumerate(file): 
        if idx > 0:
            fields = line.split(',')
            success_grasps.append((True if fields[-4][0] == 'T' else False))
            success_throws.append((True if fields[-3][0] == 'T' else False))
            explore_grasps.append((True if fields[-2][0] == 'T' else False))
            explore_throws.append((True if fields[-1][0] == 'T' else False))

print(len(success_grasps), len(success_throws), len(explore_grasps), len(explore_throws))

j = 1000
accuracy_grasp = []
accuracy_throw = []

exploit_msk = np.array([(not a and not b) for a, b in zip(explore_grasps, explore_throws)])
success_grasps = np.array(success_grasps)[exploit_msk]
success_throws = np.array(success_throws)[exploit_msk]

def accuracy(i, success):
    i_min = 0 if i < j else i-j
    accuracy = np.array(success)[i_min:i].sum() / j
    if i < j:
        accuracy = accuracy * (i / j)
    return accuracy

accuracy_grasps = []
accuracy_throws = []

for i in range(len(success_grasps)):
    accuracy_grasps.append(accuracy(i, success_grasps))
    accuracy_throws.append(accuracy(i, success_throws))

import matplotlib.pyplot as plt

plt.plot(accuracy_grasp, c='blue')
plt.show()

plt.plot(accuracy_throws, c='red')
plt.show()
