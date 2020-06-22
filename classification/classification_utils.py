def is_correct_host(diff):
    if diff ==0:
        return 1
    else:
        return 0

def get_features(features, data):
    list = []
    for f in features:
        list.append(data[f])
    X = np.vstack(list).T
    return X

# find index of array where value is nearest given value
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# score function to compute purity at fixed efficiency (=TPR)
def score_func(probs, y):
    correct = (y==1)
    wrong = (y==0)
    pur, eff, thresh = precision_recall_curve(y, probs, pos_label=1)
    purity_func = interp1d(eff[::-1], pur[::-1], kind='linear') # reverse-order so x is monotonically increasing
    metric = purity_func(0.98) # purity at fixed efficiency=98%
    return float(metric)

def score_func_CV(y,probs):
    probs = probs[:,1]
    correct = (y==1)
    wrong = (y==0)
    pur, eff, thresh = precision_recall_curve(y, probs, pos_label=1)
    purity_func = interp1d(eff[::-1], pur[::-1], kind='linear') # reverse-order so x is monotonically increasing
    metric = purity_func(0.98) # purity at fixed efficiency=98%
    return float(metric)
