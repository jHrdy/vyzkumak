from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

sys.path.append(str(parent_dir))

from ready_data import norm_data
#from ready_proj_data import norm_data_2d as norm_data

def idxs(iterable):
    return [i for i in range(len(iterable))]

svm = OneClassSVM(gamma="auto").fit(norm_data)
svm.predict(norm_data)
out_measure = svm.score_samples(norm_data)
dec = svm.decision_function(norm_data)
plt.plot(idxs(dec), dec)
plt.title("Decision function")
plt.show()

for m in range(len(out_measure)):
    # decided for decision boundary at 80 (not permanent just for testing)
    if out_measure[m] < 80:     
        plt.scatter(m,out_measure[m], color="red")
    else:
        plt.scatter(m,out_measure[m], color="blue")

plt.show()