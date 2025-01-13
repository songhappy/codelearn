import math


def get_NDCG(groundtruth, pred_rank_list, k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1) / math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(min(num_real_item, k))
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


import numpy as np
from sklearn.metrics import ndcg_score

# we have groud-truth relevance of some answers to a query:
true_relevance = np.asarray([[10, 0, 0, 1, 5]])
# we predict some scores (relevance) for the answers
scores = np.asarray([[.1, .2, .3, 4, 70]])
ndcg_sci = ndcg_score(true_relevance, scores)
print(ndcg_sci)

ndcg_my = get_NDCG([10, 0, 0, 1, 5], [.1, .2, .3, 4, 70], k=5)

print(ndcg_my)
