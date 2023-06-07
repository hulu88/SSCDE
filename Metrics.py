from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
from munkres import Munkres, print_matrix
import numpy as np

class linkpred_metrics():
    def __init__(self, edges_pos, edges_neg):
        self.edges_pos = edges_pos
        self.edges_neg = edges_neg

    def get_roc_score(self, emb, feas):
        # if emb is None:
        #     feed_dict.update({placeholders['dropout']: 0})
        #     emb = sess.run(model.z_mean, feed_dict=feed_dict)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Predict on test set of edges
        adj_rec = np.dot(emb, emb.T)
        preds = []
        pos = []
        for e in self.edges_pos:
            preds.append(sigmoid(adj_rec[e[0], e[1]]))
            pos.append(feas['adj_orig'][e[0], e[1]])

        preds_neg = []
        neg = []
        for e in self.edges_neg:
            preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            neg.append(feas['adj_orig'][e[0], e[1]])

        preds_all = np.hstack([preds, preds_neg])
        labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)

        return roc_score, ap_score, emb


import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics


def cluster_acc(y_true, y_pred):

    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)

    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')

    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')

    return acc, f1_macro, precision_macro, recall_macro


def eva(y_true, y_pred,  print_msg=True, epoch=None):

    acc, f1, precision, recall = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if print_msg:
        # print(epoch, ': acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
        #       ', f1 {:.4f}'.format(f1), ', precision {:.4f}'.format(precision), ', recall {:.4f}'.format(recall))
        print(epoch, ': acc {:.4f}'.format(acc), ', ari {:.4f}'.format(ari),', F {:.4f}'.format(f1))
    # return {'acc': np.round(acc, 4), 'nmi': np.round(nmi, 4), 'ari': np.round(ari, 4),
    #         'f1': np.round(f1, 4), 'precision': np.round(precision, 4), 'recall': np.round(recall, 4)}
    # # return {'acc': acc, 'nmi': nmi, 'ari': ari, 'f1': f1, 'precision': precision, 'recall': recall}
    # if epoch==199:
    #     fh = open('recoder.txt', 'a+')
    #     fh.write(
    #         'ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (
    #             acc, f1, precision, recall, nmi, ari))
    #     fh.write('\r\n')
    #     fh.flush()
    #     fh.close()
    return acc
