import numpy as np


def recall_k(proba, ground, k):
    recall_result = []

    top_k = np.flip(np.argsort(proba), axis=1)[:, :k]
    for y_h, y in zip(top_k, ground):
        if type(y) == list:
            recall_result.append(len(set(y) & set(y_h)) / len(y))
        else:
            recall_result.append(y in y_h)

    return np.mean(recall_result)


def precision_k(proba, ground, k):
    precision_result = []

    top_k = np.flip(np.argsort(proba), axis=1)[:, :k]
    for y_h, y in zip(top_k, ground):
        if type(y) == list:
            precision_result.append(len(set(y) & set(y_h)) / k)
        else:
            precision_result.append((y in y_h) / k)
    return np.mean(precision_result)


def f1_k(proba, ground, k):
    r_k = recall_k(proba, ground, k)
    p_k = precision_k(proba, ground, k)
    if r_k + p_k == 0:
        f1 = 0
    else:
        f1 = (2 * p_k * r_k) / (r_k + p_k)
    return f1


def dcg(rel, i):
    return rel / np.log2(i + 1)


def idcg(rel, data_length):
    accum_idcg = 0
    for i in range(1, data_length + 1):
        accum_idcg += dcg(rel, i)
    return accum_idcg


def ndcg_k(proba, ground, k):
    ndcg_result = []
    target_score = 5  # Suppose all relevance are same(Figures do not affect results)

    top_k = np.flip(np.argsort(proba), axis=1)[:, :k]
    for y_h, y in zip(top_k, ground):
        if type(y) == list:
            accum_dcg = 0
            accum_idcg = idcg(target_score, len(y))
            for ea_y in y:
                if ea_y in y_h:
                    accum_dcg += dcg(target_score, np.where(y_h == ea_y)[0][0] + 1)
        else:
            accum_dcg = 0
            accum_idcg = idcg(target_score, 1)
            if y in y_h:
                accum_dcg += dcg(target_score, np.where(y_h == y)[0][0] + 1)

        if accum_dcg == 0 or accum_idcg == 0:
            ndcg_result.append(0)
        else:
            ndcg_result.append(accum_dcg / accum_idcg)

    return np.mean(ndcg_result)
