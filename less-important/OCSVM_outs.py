from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

sys.path.append(str(parent_dir))

import plotting_styles as style
from plotting_styles import apply_global_style
from ready_data import norm_data
#from ready_proj_data import norm_data_2d as norm_data

def idxs(iterable):
    return [i for i in range(len(iterable))]

svm = OneClassSVM().fit(norm_data)
svm.predict(norm_data)
out_measure = svm.score_samples(norm_data)
dec = svm.decision_function(norm_data)
apply_global_style()
plt.plot(range(len(dec)), dec)
plt.title("Decision function")
plt.show()

plt.title("Outlier score OCSVM")
plt.scatter(range(len(out_measure)), out_measure, **style.scatter_style)
style.apply_global_style()
plt.xlabel("Histogram Index")
plt.ylabel("Score")
plt.show()
print(max(out_measure))
highest_devs = sorted(dev_dict:=enumerate(out_measure), key=lambda deviation: deviation[1], reverse=True)[-5:]
print(highest_devs[::-1])
exit()
for m in range(len(out_measure)):
    # decided for decision boundary at 80 (not permanent just for testing)
    if out_measure[m] < 80:     
        plt.scatter(m,out_measure[m], color="red")
    else:
        plt.scatter(m,out_measure[m], color="blue")

plt.show()