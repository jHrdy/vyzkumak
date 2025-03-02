from sklearn.svm import OneClassSVM
from ready_data import norm_data
import matplotlib.pyplot as plt
from LOF_outliers import idxs

svm = OneClassSVM(gamma="auto").fit(norm_data)
svm.predict(norm_data)
out_measure = svm.score_samples(norm_data)
dec = svm.decision_function(norm_data)
plt.plot(idxs(dec), dec)
plt.title("Decision function")
plt.show()

for m in range(len(out_measure)):
    if out_measure[m] < 32:
        plt.scatter(m,out_measure[m], color="red")
    else:
        plt.scatter(m,out_measure[m], color="blue")

plt.show()