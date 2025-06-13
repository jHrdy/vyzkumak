# according to CLT variable avg_distances should vector with data in normal distribution 
# question is: are the measurements (distances among points in dataset) independent (cuz I'd say they are dependent)
# let's test that!

from outliers_sklearn_knn import avg_distances_copy
from scipy.stats import kstest      # kolmogorov-smirnov test

print(kstest(avg_distances_copy, 'norm'))
# output:
# KstestResult(statistic=np.float64(0.551859729522877), 
#               pvalue=np.float64(9.542695505598672e-48), 
#               statistic_location=np.float64(0.13036135363277646),
#               statistic_sign=np.int8(-1))
# seems that data does not have normal distribution, but to be honest I am not really sure what may be the real reason 
# --> ASK SOMEONE