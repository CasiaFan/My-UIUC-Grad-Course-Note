import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def load_data(data_file, normalize=True):
    feat = []
    lbl = []
    with open(data_file, "r") as f:
        counter = 0
        for line in f:
            if counter == 0:
                counter += 1
                continue
            data = line.strip().split(",")
            feat.append([int(data[2]), int(data[3])])
            lbl.append(int(data[4]))
    feat = np.array(feat, dtype=float)
    lbl = np.array(lbl)
    if normalize:
        feat[:, 0] = feat[:, 0]/np.max(feat[:, 0])
        feat[:, 1] = feat[:, 1]/np.max(feat[:, 1]) 
    train_size = int(0.8*len(feat))
    feat_x = feat[:train_size]
    feat_y= feat[train_size:]
    lbl_x = lbl[:train_size]
    lbl_y = lbl[train_size:]
    return feat_x, lbl_x, feat_y, lbl_y

def accuracy(pred, label):
    return np.sum(pred==label)/len(pred)

def train_lr(train_x, train_y, test_x, test_y, max_iter=100, random_state=1):
    clf = LogisticRegression(max_iter=max_iter, random_state=random_state).fit(train_x, train_y)
    pred = clf.predict(test_x)
    acc = accuracy(pred, test_y)
    print("Accuracy of LR: {}".format(acc))

def train_svm(train_x, train_y, test_x, test_y, max_iter, kernel="linear", random_state=1, gamma="scale"):
    svc = svm.SVC(kernel=kernel, random_state=random_state, max_iter=max_iter, gamma=gamma).fit(train_x, train_y)
    pred = svc.predict(test_x)
    acc = accuracy(pred, test_y)
    print("Accuracy of SVM with kernel {}: {}".format(kernel, acc)) 


if __name__ == "__main__":
    data_file = "car_data.csv"
    # test 1: norm + 100 epoch
    max_iter = 100
    normalize = True
    train_x, train_y, test_x, test_y = load_data(data_file, normalize=normalize)
    # lr
    train_lr(train_x, train_y, test_x, test_y, max_iter=max_iter)
    #  linear svm
    kernel = "linear"
    train_svm(train_x, train_y, test_x, test_y, max_iter=max_iter, kernel=kernel) 
    # svm with rbf
    kernel = "rbf"
    gamma = 1
    train_svm(train_x, train_y, test_x, test_y, max_iter=max_iter, kernel=kernel, gamma=gamma)

    # test 1: norm + 100 epoch
    max_iter = 500
    normalize = False
    train_x, train_y, test_x, test_y = load_data(data_file, normalize=normalize)
    # lr
    train_lr(train_x, train_y, test_x, test_y, max_iter=max_iter)
    #  linear svm
    kernel = "linear"
    train_svm(train_x, train_y, test_x, test_y, max_iter=max_iter, kernel=kernel) 
    # svm with rbf
    kernel = "rbf"
    gamma = 1
    train_svm(train_x, train_y, test_x, test_y, max_iter=max_iter, kernel=kernel, gamma=gamma)


    