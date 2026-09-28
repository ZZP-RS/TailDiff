# -*- coding: utf-8 -*-

import numpy as np
import math
import random
import os
import json
from scipy.sparse import dok_matrix, csr_matrix
import torch
from math import radians, cos, sin, sqrt, atan2
from datetime import datetime
from dateutil import parser

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n - r)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def cosine_similarity(matrix):
    # 计算向量的L2范数（长度）
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 对矩阵进行归一化，使得每个向量的范数为1
    normalized_matrix = matrix / norms
    # 计算归一化后矩阵的内积，得到相似度矩阵
    similarity_matrix = np.matmul(normalized_matrix, normalized_matrix.T)
    return similarity_matrix


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=200, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.score_min = None
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        # Saves model when validation loss decrease.
        
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


# ==== 加载 rich user 序列 ====
# ==== 加载 rich user 序列 ====
def get_user_seqs_rich(json_path):
    """
    从包含时间与地理信息的 JSON 序列文件中加载用户交互数据。
    格式示例：{
      "0": [
        {"item": 123, "time": "2019-01-01T08:00:00Z", "lat": 30.2, "lon": 120.1},
        ...
      ],
      ...
    }
    """
    with open(json_path, "r") as f:
        user_dict = json.load(f)
    rich_seqs = []
    for uid in sorted(user_dict, key=lambda x: int(x)):
        rich_seqs.append(user_dict[uid])
    return rich_seqs
# ==== 计算两点地理距离（单位：km） ====
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def parse_time(time_str, dataset_name):
    """根据数据集名称解析时间字符串，增加类型检查与容错"""
    
    # 1. 自动处理 bytes 类型（解决 b'Fri ' 报错的关键）
    if isinstance(time_str, bytes):
        time_str = time_str.decode('utf-8')
    
    # 2. 去除两端空格
    time_str = time_str.strip()

    # 3. 统一转为小写进行匹配
    ds_name = dataset_name.lower()

    if ds_name in ["tky", "nyc", "10tky"] or "fri" in time_str.lower():
        # 格式: 'Fri Aug 10 06:11:31 +0000 2012'
        try:
            # 尝试带时区的解析
            return datetime.strptime(time_str, '%a %b %d %H:%M:%S %z %Y')
        except ValueError:
            # 如果没有时区偏移（例如没有 +0000），尝试不带时区的格式
            # 或者直接使用通用的 parser.parse
            return parser.parse(time_str)
    else:
        # 针对 '2019-01-01T08:00:00Z' 等 ISO 格式
        try:
            return parser.isoparse(time_str)
        except (ValueError, TypeError):
            # 万一识别失败，兜底方案：使用万能解析器
            return parser.parse(time_str)


# ==== 构建 soft mask 评分矩阵（训练） ====
def generate_soft_mask_matrix_valid(rich_seqs, num_users, num_items, data_name, geo_weight=0.5, time_weight=0.5):
    """
    构建带时间与地理惩罚权重的训练评分矩阵（排除最近两个交互）
    """
    rating_matrix = dok_matrix((num_users, num_items), dtype=np.float32)
    for uid, seq in enumerate(rich_seqs):
        for i in range(len(seq) - 2):
            poi_info = seq[i]
            poi_id = poi_info["item"]
            if poi_id >= num_items:
                continue
            score = 1.0
            if i > 0:
                last = seq[i-1]
                # 计算距离（确保非负，避免负数导致 exp 溢出）
                distance = haversine(last["lat"], last["lon"], poi_info["lat"], poi_info["lon"])
                distance = max(distance, 0.0)  # 强制距离非负
                
                # 计算时间差（确保非负）
                time_current = parse_time(poi_info["time"], data_name)
                time_last = parse_time(last["time"], data_name)
                time_diff = (time_current - time_last).total_seconds() / 3600.0
                time_diff = max(time_diff, 0.0)  # 强制时间差非负
                
                # 根据数据集设置衰减参数，并限制 exp 输入的最大值（避免溢出）
                if data_name.lower() in ["tky", "nyc", "10tky"]:
                    # 限制 distance 最大为 20（超过则按 20 计算，避免 exp(-distance/1.0) 下溢，但主要防止负数导致的上溢）
                    distance_clamped = min(distance, 20.0)
                    geo_decay = np.exp(-distance_clamped / 1.0)
                    
                    # 限制 time_diff 最大为 24（超过则按 24 计算）
                    time_diff_clamped = min(time_diff, 24.0)
                    time_decay = np.exp(-time_diff_clamped / 3.0)
                elif data_name.lower() in ["new_orleans", "philadelphia"]:
                    distance_clamped = min(distance, 100.0)  # 对应分母 10.0，适当放大上限
                    geo_decay = np.exp(-distance_clamped / 10.0)
                    
                    time_diff_clamped = min(time_diff, 120.0)  # 对应分母 12.0
                    time_decay = np.exp(-time_diff_clamped / 12.0)
                
                # 对 exp 结果设置上限（进一步防止溢出）
                geo_decay = np.clip(geo_decay, 0.0, 1e6)  # 限制最大值为 1e6（远大于合理值）
                time_decay = np.clip(time_decay, 0.0, 1e6)
                
                score = geo_weight * geo_decay + time_weight * time_decay
                score = max(score, 0.1)  # 确保最低分，避免过小值
                
                # 限制 score 最大值，避免后续矩阵转换溢出
                score = min(score, 1e6)
                
            rating_matrix[uid, poi_id] += score
    return rating_matrix.tocsr()

# ==== 构建 soft mask 评分矩阵（测试） ====
def generate_soft_mask_matrix_test(rich_seqs, num_users, num_items, data_name, geo_weight=0.5, time_weight=0.5):
    """
    构建带时间与地理惩罚权重的测试评分矩阵（排除最近一个交互）
    """
    rating_matrix = dok_matrix((num_users, num_items), dtype=np.float32)
    for uid, seq in enumerate(rich_seqs):
        for i in range(len(seq) - 1):
            poi_info = seq[i]
            poi_id = poi_info["item"]
            if poi_id >= num_items:
                continue
            score = 1.0
            if i > 0:
                last = seq[i-1]
                # 确保距离非负
                distance = haversine(last["lat"], last["lon"], poi_info["lat"], poi_info["lon"])
                distance = max(distance, 0.0)
                
                # 确保时间差非负
                time_current = parse_time(poi_info["time"], data_name)
                time_last = parse_time(last["time"], data_name)
                time_diff = (time_current - time_last).total_seconds() / 3600.0
                time_diff = max(time_diff, 0.0)
                
                # 限制输入到 exp 的值
                geo_decay = np.exp(-min(distance, 100.0) / 10.0)  # 对应分母 10.0，上限 100
                time_decay = np.exp(-min(time_diff, 120.0) / 12.0)  # 对应分母 12.0，上限 120
                
                # 限制 exp 结果最大值
                geo_decay = np.clip(geo_decay, 0.0, 1e6)
                time_decay = np.clip(time_decay, 0.0, 1e6)
                
                score = geo_weight * geo_decay + time_weight * time_decay
                # 限制 score 最大值
                score = min(score, 1e6)
            
            rating_matrix[uid, poi_id] += score
    return rating_matrix.tocsr()

def get_threshold_value(data_file):
    # 计算阈值：长尾用户的长度与用户交互长度低于这个阈值的就是长尾用户
    file = open(data_file)
    lines = file.readlines()
    user_lengths = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        user_lengths.append(len(items))
    sum_lengths = sum(user_lengths)
    threshold_value = int(0.2 * sum_lengths)
    total_sum = 0
    threshold = 0
    for i in sorted(user_lengths):
        if total_sum < threshold_value:
            total_sum += i
        else:
            threshold = i
            print(threshold)
            break
    return threshold


'''def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 1

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix'''


def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence


def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set)  # 331
    return item2attribute, attribute_size


def get_metric(pred_list, topk=10):
    ndcg = 0.0
    hit = 0.0
    mrr = 0.0
    for rank in pred_list:
        mrr += 1.0 / (rank + 1.0)
        if rank < topk:
            ndcg += 1.0 / np.log2(rank + 2.0)
            hit += 1.0
    return hit / len(pred_list), ndcg / len(pred_list), mrr / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


'''def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users'''

def recall_at_k(actual, predicted, topk):
    hits = 0
    num_users = len(predicted)

    for i in range(num_users):
        gt_item = actual[i]  # 单个item，不是list
        pred_topk = predicted[i][:topk]

        if gt_item in pred_topk:
            hits += 1

    return hits / num_users
def apk(actual, predicted, k=10):
    """
        Computes the average precision at k.
        This function computes the average precision at k between two lists of
        items.
        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements
        Returns
        -------
        score : double
                The average precision at k over the input lists
        """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def read_user_interactions(file_path):
    user_interactions = {}

    with open(file_path, 'r') as file:
        for line in file:
            items = list(map(int, line.strip().split()[:]))
            user_id = items[0]
            user_interactions[user_id] = items[1:]
    return user_interactions


def check_user_sequence(user_seq, user_interactions, value):
    filtered_seq = [x for x in user_seq.tolist() if x != 0]

    for interactions in user_interactions.values():
        idx1, idx2 = len(filtered_seq) - 1, len(interactions) - 2
        while idx1 >= 0 and idx2 >= 0:
            if filtered_seq[idx1] == interactions[idx2]:
                idx1 -= 1
                idx2 -= 1
            else:
                break

        if idx1 == -1 and len(interactions) < value:
            return True
    return False
