import math
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
from hnswlib import Index
import torch
import torch.nn as nn
from modules import Encoder, LayerNorm
from utils import cosine_similarity
import json
from torch.nn.parameter import Parameter
from torch.nn.functional import softmax


class SASRecModel(nn.Module):
    def __init__(self, args):
        super(SASRecModel, self).__init__()
        self.args = args

        # ====== 👑 彻底简化：直接用 args 传入的地址读取三模态 🟢 ======
        img_path = args.img_embedding_path
        txt_path = args.txt_embedding_path
        meta_path = args.meta_embedding_path

        if os.path.exists(img_path) and os.path.exists(txt_path) and os.path.exists(meta_path):
            print(f"[Info] 正在加载三模态 768 维原始特征...")
            img_emb = torch.tensor(np.load(img_path), dtype=torch.float)
            txt_emb = torch.tensor(np.load(txt_path), dtype=torch.float)
            meta_emb = torch.tensor(np.load(meta_path), dtype=torch.float)

            # 1. 读入原始 768 维矩阵，设置 freeze=True，不破坏大模型提取出来的原始特征
            self.img_embeddings = nn.Embedding.from_pretrained(img_emb, freeze=True, padding_idx=0)
            self.txt_embeddings = nn.Embedding.from_pretrained(txt_emb, freeze=True, padding_idx=0)
            self.meta_embeddings = nn.Embedding.from_pretrained(meta_emb, freeze=True, padding_idx=0)

            # 2. 🟢 核心新增：定义可学习的线性降维层，自动将 768 维动态映射压缩到推荐系统专用的 128 维（args.hidden_size）
            self.img_proj = nn.Linear(img_emb.shape[1], args.hidden_size)
            self.txt_proj = nn.Linear(txt_emb.shape[1], args.hidden_size)
            self.meta_proj = nn.Linear(meta_emb.shape[1], args.hidden_size)

            self.img_ln = nn.LayerNorm(args.hidden_size)
            self.txt_ln = nn.LayerNorm(args.hidden_size)
            self.meta_ln = nn.LayerNorm(args.hidden_size)

            # 3. 💡 定义可学习的注意力权重参数 w1, w2, w3（伴随降维层在 Stage 1 协同训练）
            self.modal_weights = nn.Parameter(torch.ones(3, dtype=torch.float))
            self.use_multimodal_fusion = True
            print(f"[Success] 768维->{args.hidden_size}维 降维投影与可学习融合模型构建成功！")
        else:
            print("[Warning] 未找到完整的 .npy 文件，退化为普通随机初始化单表")
            self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
            self.use_multimodal_fusion = False
        # ======================================================================

        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.apply(self.init_weights)
    # ====== 👑 新增：融合公式函数（内含降维与相加） 👑 ======
    def get_multimodal_embeddings(self, item_ids):
        """
        实现公式: e_i = w1 * Proj(e_img) + w2 * Proj(e_txt) + w3 * Proj(e_meta)
        返回维度: [Batch_size, Seq_len, 128] 或 [Item_size, 128]
        """
        if not self.use_multimodal_fusion:
            return self.item_embeddings(item_ids)

        # ==============================================================================
        # 🟢 [核心修改] 引入温度系数机制，欺骗 Softmax 放大参数差异
        # ==============================================================================
        # 1. 设定温度系数（推荐 0.05）。它能把底层的微小差距（如 0.01）强行放大 20 倍变成 0.2，从而让 Softmax 敏感起来
        temperature = 0.05 

        # 2. 对原始参数除以温度系数后再进行 Softmax 激活
        w = torch.softmax(self.modal_weights / temperature, dim=0)
        w1, w2, w3 = w[0], w[1], w[2]

        # 3. [内部监控] 1% 的概率在训练时打印宏观上真正被拉开后的有效权重，验证是否打破 0.33
        
        # ==============================================================================

        # 4. 从各自表里提取 768 维特征
        e_img_raw = self.img_embeddings(item_ids)
        e_txt_raw = self.txt_embeddings(item_ids)
        e_meta_raw = self.meta_embeddings(item_ids)

        # 5. 通过可学习的线性层降维至 128 维
        e_img = self.img_ln(self.img_proj(e_img_raw))
        e_txt = self.txt_ln(self.txt_proj(e_txt_raw))
        e_meta = self.meta_ln(self.meta_proj(e_meta_raw))

        # 6. 加权交融成最终的高阶语义嵌入
        e_combined = w1 * e_img + w2 * e_txt + w3 * e_meta
        return e_combined

    # 🟢 请在 models.py 的 SASRecModel 类中，新增或替换这个动态属性：
   # ======= 🟢 1. 动态属性：供 Trainer 实时获取激活后的三模态权重 =======
    @property
    def get_modal_weights(self):
        """
        返回经过温度系数（0.05）Softmax 激活后的真实有效三模态权重 (w_img, w_txt, w_meta)
        """
        with torch.no_grad():
            temperature = 0.05
            w = torch.softmax(self.modal_weights / temperature, dim=0)
            return [round(x.item(), 4) for x in w]

    # ======= 🟢 2. 动态属性：每次访问自动重算全表（去掉普通 def，统一用 property，这样调用就不用加括号了） =======
    @property
    def item_embeddings_table(self):
        """
        硬核重构：砸碎死表！
        每次外部调用 self.model.item_embeddings_table 时，都会动态实时计算全量多模态交融特征。
        """
        # 生成从 0 到 item_size-1 的全量物品 ID 向量
        all_item_ids = torch.arange(self.args.item_size, device=self.modal_weights.device)
        
        # 实时调用多模态融合函数，此时计算图是活的、相连的！
        return self.get_multimodal_embeddings(all_item_ids)

    def forward(self, input_ids):
        # 🟢 查表替换为动态融合降维调用
        items_embeddings = self.get_multimodal_embeddings(input_ids)

        sequence_length = input_ids.size(1)
        position_ids = torch.arange(sequence_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        sequence_emb = items_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        attention_mask = (input_ids > 0).long().unsqueeze(1).unsqueeze(2)
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(input_ids.device)
        attention_mask = attention_mask * subsequent_mask
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        if hasattr(self.args, 'num_hidden_layers'):
            extended_attention_mask = attention_mask
            item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
            sequence_output = item_encoded_layers[-1]
        else:
            sequence_output = self.item_encoder(sequence_emb, attention_mask)

        return sequence_output
    # Positional Embedding
    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # model same as SASRec
    def transformer_encoder(self, input_ids):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class OfflineItemSimilarity:
    def __init__(self, args, data_file=None, similarity_path=None, model_name='ItemCF', \
                 dataset_name='ml-1m'):

        self.args = args
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        

        """
            train_data_list:
            [[],[],[]],每个[]为不含后3个item的序列

            train_data将每个用户喜欢的物品转化为一个列表
            [('1', '1', 1), ('1', '2', 1), ('1', '3', 1), ('1', '4', 1), ('1', '5', 1)]
            ()中第一个代表userId
            第二个代表itemId
            第三个每个都是1,表示喜欢

            train_data_dict
            数据为{'1': {'1': 1, '2': 1, '3': 1, '4': 1, '5': 1}, 
            '2': {'9': 1, '10': 1, '11': 1}}
            表示用户1对物品1态度为1(喜欢)
            调用方法为train_data[1][1]
         """


        # 选取模型
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        self.data_path = data_file

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path='./similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict

        C = dict()
        N = dict()  # N统计了每个物品出现的次数

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                """
                    idx为迭代次数从0开始,u为用户,
                    items格式为{'9': 1, '10': 1, '11': 1} '物品9':喜欢
                """

                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        """
                        i为物品编号
                        setdefault(key[, default]),如果没有 key,会加入这个key,值设为0
                        有这个key,直接返回字典中对应的key 的值    
                        """
                        N.setdefault(i, 0)
                        N[i] += 1  # N统计了每个物品出现的次数
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            """
                itemSimBest 是一个嵌套字典，其中第一层的键表示当前物品的编号，
                第二层的键表示与当前物品相似的其他物品的编号，
                第二层的值则表示这些相似物品与当前物品之间的相似度分数。
            """
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0);
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)

        elif self.model_name == 'cosine_similarity':
            item_embed = np.load("/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/fused_embedding_pca128.npy")
            print(f"Embedding shape: {item_embed.shape}")  # 应该是 (num_items, dim)

            self.args.item_size = item_embed.shape[0]
            item_similarity_matrix = cosine_similarity(item_embed)

            self.itemSimBest = {}
            for i in range(item_similarity_matrix.shape[0]):
                self.itemSimBest[i] = {}
                for j in range(item_similarity_matrix.shape[1]):
                    if i == j:
                        continue
                    self.itemSimBest[i][j] = item_similarity_matrix[i, j]

            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'HNSW':
            # 创建 HNSW 索引
            item_embed = np.load("/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/fused_embedding_pca128.npy")
            self.args.item_size = item_embed.shape[0]
            hnsw_index = Index(space='cosine', dim=item_embed.shape[1])
            hnsw_index.init_index(max_elements=item_embed.shape[0], ef_construction=2000, M=64)
            hnsw_index.add_items(item_embed)

            # 计算物品之间的相似度
            itemSimBest = {}
            for i in range(item_embed.shape[0]):
                itemSimBest[i] = {}
                neighbors, _ = hnsw_index.knn_query(item_embed[i], k=item_embed.shape[0])
                for j, score in zip(neighbors[0], _[0]):
                    if i == j:
                        continue
                    itemSimBest[i][j] = 1 - score

            self._save_dict(itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")

            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'cosine_similarity', 'HNSW']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'cosine_similarity', 'HNSW']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(), key=lambda x: x[1], \
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (float(x[1]) - self.min_score) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(), key=lambda x: x[1], \
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (float(x[1]) - self.min_score) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))



class MultiModalSimilarity:
    """专门处理三模态(图像、评论、元数据)相似度融合的类"""
    def __init__(self, 
                 image_sim_path: str, 
                 text_sim_path: str, 
                 meta_sim_path: str,
                 poi2id_path: str,
                 args=None,
                 device: str = 'cuda',
                 enabled_modals: list = ['image', 'text', 'meta']  # 新增参数：控制启用的模态
                 ):
        """
        参数说明:
            image_sim_path: 图像模态相似度矩阵路径(.npy)
            text_sim_path: 评论模态相似度矩阵路径(.npy) 
            meta_sim_path: 元数据模态相似度矩阵路径(.npy)
            poi2id_path: POI到ID的映射文件路径(.json)
            device: 计算设备(cpu/cuda)
        """
        self.device = device
        self.enabled_modals = enabled_modals  # 保存启用的模态
        self.args = args
        # 根据启用的模态数量初始化权重（默认各占1/len(enabled_modals)）
        # ========== 修改点1：修正可学习权重初始化（适配启用的模态数量） ==========
        # 用nn.Parameter正确初始化，确保梯度开启
        self.alphas = nn.Parameter(
            torch.ones(len(self.enabled_modals), device=device, dtype=torch.float32)  # 初始值1，保证梯度
        )
       
        
        # 加载POI映射
        with open(poi2id_path, 'r') as f:
            self.poi2id = json.load(f)
            self.id2poi = {int(v): k for k, v in self.poi2id.items()}
        
        # 加载三种模态的相似度矩阵
        # 只加载启用的模态的相似度矩阵
        self.sim_matrices = {}
        if 'image' in enabled_modals:
            self.sim_matrices['image'] = self._load_sparse_matrix(image_sim_path)
        if 'text' in enabled_modals:
            self.sim_matrices['text'] = self._load_sparse_matrix(text_sim_path)
        if 'meta' in enabled_modals:
            self.sim_matrices['meta'] = self._load_sparse_matrix(meta_sim_path)

        # 在 models.py 约 100 行左右，加载完矩阵后
        for modal_name, modal_dict in self.sim_matrices.items():
            first_key = list(modal_dict.keys())[0]
            # 检查字典里 Key 的最大值和最小值
            keys = list(modal_dict.keys())
            print(f"DEBUG: 模态 {modal_name} 的 ID 范围是: {min(keys)} 到 {max(keys)}")
        if self.args is not None:
            print(f"DEBUG: 当前模型的 item_size 是: {self.args.item_size}")
        else:
            print("DEBUG: 警告，args 传入为 None")
        # 计算最大最小相似度用于归一化
        self.max_score, self.min_score = self._get_sim_range()

    def get_weighted_similar(self, item_id, top_k=1):
        """
        真正使用可学习权重融合多模态相似度（修复原代码的错误）
        权重会参与梯度传播！
        参数:
            item_id: 物品内部ID（整数）
            top_k: 返回相似物品数量
        返回:
            相似物品ID列表（排除自身）
        """
        # 1. 校验item_id是否存在
        if item_id not in self.sim_matrices[self.enabled_modals[0]]:
            print(f"WARN: item_id {item_id} 不在相似度矩阵中")
            return []
        
        # 2. 对权重做softmax（保证权重和为1，且可导）
        weights = softmax(self.alphas, dim=0)  # 形状: [启用的模态数]
        
        # 3. 初始化融合相似度字典（存储当前item与所有其他item的融合相似度）
        fused_sim_dict = {}
        
        # 4. 遍历所有启用的模态，累加加权后的相似度
        for idx, modal_name in enumerate(self.enabled_modals):
            modal_sim_dict = self.sim_matrices[modal_name]
            if item_id not in modal_sim_dict:
                continue
            # 遍历当前模态下item_id的所有相似项
            for related_item, sim_score in modal_sim_dict[item_id].items():
                if related_item not in fused_sim_dict:
                    fused_sim_dict[related_item] = 0.0
                # 加权累加（核心：可学习权重 * 模态相似度）
                fused_sim_dict[related_item] += weights[idx].item() * sim_score
        
        # 5. 转换为tensor并取topk（保留梯度）
        # 提取item列表和相似度分数
        related_items = list(fused_sim_dict.keys())
        sim_scores = torch.tensor([fused_sim_dict[item] for item in related_items], 
                                 device=self.device, dtype=torch.float32)
        
        # 取top_k+1（排除自身）
        top_k_scores, top_k_indices = torch.topk(sim_scores, k=min(top_k+1, len(sim_scores)))
        
        # 6. 转换为ID列表并排除自身
        top_k_item_ids = [related_items[idx] for idx in top_k_indices.cpu().numpy()]
        top_k_item_ids = [idx for idx in top_k_item_ids if idx != item_id][:top_k]
        
        return top_k_item_ids
    def _load_sparse_matrix(self, path: str) -> dict:
        """
        加载numpy数组格式的相似度矩阵，并转换为嵌套字典格式
        {id_a: {id_b: 相似度分数}}
        只保留非零值以节省内存（如果矩阵是稀疏的）
        """
        # 加载二维numpy数组 (shape: [num_items, num_items])
        matrix = np.load(path, allow_pickle=True)
        num_items = matrix.shape[0]
        
        
        # 转换为嵌套字典 {id_a: {id_b: score}}
        sim_dict = {}
        for id_a in range(num_items):
            # 为当前id_a创建子字典
            sim_dict[id_a] = {}
            # 遍历所有可能的id_b
            for id_b in range(num_items):
                score = matrix[id_a][id_b]
                # 只保留非零相似度（如果矩阵是稀疏的，可大幅减少内存占用）
                if score > 0 and not np.isclose(score, 0.0):
                    sim_dict[id_a][id_b] = float(score)
        # 检查加载的矩阵是否有非零值
        non_zero_count = sum(len(v) for v in sim_dict.values())
        print(f"从 {path} 加载的相似度矩阵非零值数量: {non_zero_count}")
        if non_zero_count == 0:
            print(f"警告：{path} 中未找到有效相似度数据")
        
        return sim_dict

    def _get_sim_range(self) -> tuple:
        """计算所有模态的最大最小相似度"""
        max_score, min_score = -1, 2  # 初始化超出合理范围的值
        
        for modal in self.sim_matrices.values():
            for i in modal:
                for j in modal[i]:
                    score = modal[i][j]
                    max_score = max(max_score, score)
                    min_score = min(min_score, score)
        
        # 如果所有相似度相同，避免除以零
        if max_score == min_score:
            max_score += 0.01
        return max_score, min_score

    def normalize_sim(self, score: float) -> float:
        """将相似度归一化到[0,1]区间"""
        return (score - self.min_score) / (self.max_score - self.min_score)
    def precompute_topk(self, top_k=50):
        """预先计算所有物品的 Top-K 相似物品，存入 dict"""
        self.topk_cache = {}
        # 获取物品总数，通常 id2poi 包含了所有有效的物品映射
        item_count = len(self.id2poi) 
    
        print(f"正在为 {item_count} 个物品预计算多模态相似度矩阵...")
    
        # 遍历所有内部 ID
        for internal_id in tqdm(range(item_count)):
            # 调用类中已有的 most_similar 方法
            # 这里的 most_similar 内部会处理 id2poi 的转换
            # 注意：如果 internal_id 是从 0 开始的连续整数，直接 range 没问题
            self.topk_cache[internal_id] = self.most_similar(internal_id, top_k=top_k)
    
        print("预计算完成！")


    def get_fused_similarity(self, poi_a: str, poi_b: str) -> float:
        """
        计算两个POI之间的融合相似度
        参数:
            poi_a: POI ID(字符串)
            poi_b: POI ID(字符串)
        返回:
            融合后的相似度(0-1之间)
        """
        # 转换为内部ID
        id_a = self.poi2id.get(poi_a, -1)
        id_b = self.poi2id.get(poi_b, -1)
        if id_a == -1 or id_b == -1:
            return 0.0
        
        # 只获取启用的模态的相似度
        sims = []
        for modal in self.enabled_modals:
            sim = self.sim_matrices[modal].get(id_a, {}).get(id_b, 0)
            sims.append(sim)
    
        # 基于启用的模态权重计算融合相似度
        weights = softmax(self.alphas, dim=0)  # 权重自动适配启用的模态数量
        fused_sim = sum(w * s for w, s in zip(weights, sims))
    
        return self.normalize_sim(fused_sim)
    '''def most_similar(self, item, top_k=1, with_score=False):
        """
        兼容数据增强逻辑的相似物品查询接口
        参数:
            item: 输入的POI ID（字符串或整数，根据数据格式调整）
            top_k: 返回的相似物品数量
            with_score: 是否返回相似度分数
        返回:
            若with_score=True，返回 [(相似POI ID, 分数), ...]
            否则返回 [相似POI ID, ...]
        """
        try:
        # 处理两种可能的ID格式：原始字符串ID / 数字简化ID
            item_str = str(item)
            internal_id = None
        
            # 情况1：item是原始字符串ID（和poi2id的key一致）
            if item_str in self.poi2id:
                internal_id = self.poi2id[item_str]
            # 情况2：item是数字简化ID（需要反查id2poi，找到原始字符串ID）
            else:
                # 先尝试把item转成整数，再查id2poi（id2poi是{内部整数ID: 原始字符串ID}）
                try:
                    item_int = int(item_str)
                    if item_int in self.id2poi:  # 验证这个内部ID是否存在
                        internal_id = item_int
                except ValueError:
                    pass
        
            if internal_id is None:
                print(f"获取相似物品失败: {item}（原因：ID格式不匹配，未找到对应映射，错误码：1003）")
                return []
        
            # 直接用内部ID获取相似物品（跳过原始ID转换，提高效率）
            candidates = set()
            for modal in self.sim_matrices.values():
                if internal_id in modal:
                    candidates.update(modal[internal_id].keys())
        
            if not candidates:
                print(f"获取相似物品失败: {item}（原因：无相似物品，错误码：1002）")
                return []
        
            # 计算相似度（复用之前的逻辑）
            weights = torch.softmax(self.alphas, dim=0)  # 形状：[3]（带梯度）
            scores = []
            for candidate_id in candidates:
                if candidate_id == internal_id:
                    continue
                modal_sims = []
                for modal in self.enabled_modals:
                    sim = self.sim_matrices[modal].get(internal_id, {}).get(candidate_id, 0.0)
                    modal_sims.append(sim)
                fused_sim = sum(w * s for w, s in zip(weights, modal_sims))
                normalized_sim = self.normalize_sim(fused_sim)
                scores.append((candidate_id, normalized_sim))
        
            # 排序并返回结果（返回原始字符串ID，适配后续逻辑）
            top_items = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
            if with_score:
                return [(self.id2poi[item_id], score) for item_id, score in top_items]
            return [self.id2poi[item_id] for item_id, _ in top_items]
    
        except Exception as e:
            print(f"获取相似物品失败: {item}（原因：{str(e)}，错误码：1003）")
            return []'''
    '''def most_similar(self, item, top_k=1, with_score=False):
        try:
            internal_id = None
        
            # 1. 尝试识别 internal_id (模型内部连续索引)
            # 既然 log 显示矩阵 Key 是 int，我们优先保证拿到 int 类型的 ID
            try:
                # 如果传入的是数值（可能是 170 这种内部 ID）
                item_int = int(item)
                # 这里的判断标准：只要它在任何一个相似度矩阵里，它就是有效的 internal_id
                for modal in self.sim_matrices.values():
                    if item_int in modal:
                        internal_id = item_int
                        break
            except (ValueError, TypeError):
                pass

            # 2. 如果第一步失败，尝试通过 poi2id 映射查找
            if internal_id is None:
                item_str = str(item)
                if item_str in self.poi2id:
                    internal_id = self.poi2id[item_str]
        
            # 3. 如果依然找不到映射，说明该物品确实不在相似度库中
            if internal_id is None:
                return [item] if not with_score else [(item, 1.0)]
            # 【新增】：获取当前模型允许的最大索引（由 args 传入，Philadelphia 为 2966）
            # 如果 args 没传进来，默认不限制（兼容旧逻辑）
            max_item_id = self.args.item_size if self.args is not None else float('inf')
            # 2. 计算融合相似度（带可学习权重）
            weights = softmax(self.alphas, dim=0)
            sim_dict = {}
            # 遍历所有启用的模态
            for idx, modal_name in enumerate(self.enabled_modals):
                modal_sim = self.sim_matrices[modal_name].get(internal_id, {})
                for rel_item, score in modal_sim.items():
                    if rel_item not in sim_dict:
                        sim_dict[rel_item] = 0.0
                    sim_dict[rel_item] += weights[idx].item() * score
            
            # 3. 过滤自身 + 排序
            # 3. 过滤 + 排序
            # 【核心修改】：在过滤自身的同时，过滤掉所有 >= max_item_id 的越界 ID
            sim_list = [
                (k, self.normalize_sim(v)) 
                for k, v in sim_dict.items() 
                if k != internal_id and k < max_item_id
            ]
            sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)[:top_k]
            
            # 4. 转换为原始POI ID + 返回
            if with_score:
                return [(self.id2poi.get(k, k), v) for k, v in sim_list]
            else:
                return [self.id2poi.get(k, k) for k, v in sim_list]
        except Exception as e:
            print(f"ERROR: 获取相似物品失败 {item} - {str(e)}")
            return [] if not with_score else []'''

    def most_similar(self, item, top_k=1, with_score=False):
        try:
            internal_id = None
        
            # 1. 尝试识别 internal_id
            try:
                item_int = int(item)
                # 检查是否在任一矩阵中
                for modal in self.sim_matrices.values():
                    if item_int in modal:
                        internal_id = item_int
                        break
            except (ValueError, TypeError):
                pass

            # 2. 如果第一步失败，尝试映射
            if internal_id is None:
                item_str = str(item)
                if item_str in self.poi2id:
                    internal_id = self.poi2id[item_str]
        
            # 【重要补丁】：如果找不到 ID 或 ID 越界，直接返回原物品 ID，避免产生非法索引
            max_legal_id = self.args.item_size if self.args is not None else 2966
            
            if internal_id is None or internal_id >= max_legal_id:
                return [item] if not with_score else [(item, 1.0)]

            # 3. 计算融合相似度
            weights = softmax(self.alphas, dim=0)
            sim_dict = {}
            for idx, modal_name in enumerate(self.enabled_modals):
                modal_sim = self.sim_matrices[modal_name].get(internal_id, {})
                for rel_item, score in modal_sim.items():
                    # 【核心过滤】：只处理在 Embedding 范围内的相似项
                    if rel_item < max_legal_id:
                        if rel_item not in sim_dict:
                            sim_dict[rel_item] = 0.0
                        sim_dict[rel_item] += weights[idx].item() * score
            
            # 4. 过滤自身 + 排序
            sim_list = [(k, self.normalize_sim(v)) for k, v in sim_dict.items() if k != internal_id]
            sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)[:top_k]
            
            # 【兜底】：如果过滤后没东西了，用原物品自己
            if not sim_list:
                return [item] if not with_score else [(item, 1.0)]

            # 5. 转换为原始POI ID + 返回
            if with_score:
                return [(self.id2poi.get(k, k), v) for k, v in sim_list]
            else:
                return [self.id2poi.get(k, k) for k, v in sim_list]
                
        except Exception as e:
            # 任何报错都返回原项，保证程序不崩
            return [item] if not with_score else [(item, 1.0)]
            
    def get_similar_items(self, 
                         poi_id: str, 
                         top_k: int = 10, 
                         return_scores: bool = False):
        """
        获取与指定POI最相似的其他POI
        参数:
            poi_id: 查询POI的ID
            top_k: 返回的最相似项目数
            return_scores: 是否返回相似度分数
        返回:
            相似POI列表(带或不带分数)
        """
        internal_id = self.poi2id.get(poi_id, -1)
        
        if internal_id == -1:
            print(f"POI {poi_id} 不在poi2id映射中")
            return [] if not return_scores else []
         # 计算当前模态权重（softmax归一化）
        weights = softmax(self.alphas, dim=0).detach().cpu().numpy()  # 动态权重
        
        
        # 收集所有候选POI(合并三种模态的邻居)
        candidates = set()
        for modal in self.sim_matrices.values():
            if internal_id in modal:
                candidates.update(modal[internal_id].keys())
        
        
        # 计算融合相似度
        scores = []
        for candidate_id in candidates:
            if candidate_id == internal_id:
                continue  # 排除自身
             # 计算各模态相似度
            modal_sims = []
            for idx, modal in enumerate(self.enabled_modals):
                sim = self.sim_matrices[modal].get(internal_id, {}).get(candidate_id, 0.0)
                modal_sims.append(sim)

            # 加权融合（动态权重）
            fused_sim = sum(w * s for w, s in zip(weights, modal_sims))
        
            normalized_sim = self.normalize_sim(fused_sim)  # 归一化到[0,1]
            scores.append((candidate_id, normalized_sim))
        
        # 按相似度排序并取Top-K
        top_items = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        if return_scores:
            return [(self.id2poi[item_id], score) for item_id, score in top_items]
        return [self.id2poi[item_id] for item_id, _ in top_items]

    def get_alpha_weights(self) -> dict:
        """获取当前各模态的权重(softmax后)"""
        weights = softmax(self.alphas, dim=0).detach().cpu().numpy()
        return {
            'image_weight': float(weights[0]),
            'text_weight': float(weights[1]),
            'meta_weight': float(weights[2])
        }

# =========================================================================
# =============== 新增：基于一维 Transformer 的轨迹重构扩散模型 ===============
# =========================================================================
import math

class TimestepEmbedding(nn.Module):
    """为扩散时间步 t 提供标准的正弦位置编码"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, timesteps):
        half_dim = self.hidden_size // 2
        exponent = -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / half_dim
        embeddings = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        if self.hidden_size % 2 == 1:
            embeddings = nn.functional.pad(embeddings, (0, 1))
        return embeddings

class DiTBlock(nn.Module):
    """一维 AdaLN (Adaptive Layer Normalization) Transformer 块"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        # AdaLN 映射：动态预测两个残差块各自的 scale 和 shift 参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 4)
        )

    def forward(self, x, cond_emb):
        mod = self.adaLN_modulation(cond_emb)
        shift_msa, scale_msa, shift_mlp, scale_mlp = torch.chunk(mod, 4, dim=-1)

        # 调控第一部分: 自注意力机制 (Self-Attention)
        x_norm = self.layernorm1(x)
        x_modulated = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + attn_out

        # 调控第二部分: 前馈网络 (MLP)
        x_norm = self.layernorm2(x)
        x_modulated = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + self.mlp(x_modulated)
        return x

class TrajectoryDiT(nn.Module):
    """核心 DiT 去噪网络"""
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size
        
        # 1. 时间步编码
        self.t_embed = nn.Sequential(
            TimestepEmbedding(args.hidden_size),
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.SiLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )
        # 2. 双条件（时序预测特征 + 多模态全局语义兴趣）融合投影
        self.cond_fusion = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.SiLU(),
            nn.Linear(args.hidden_size, args.hidden_size)
        )
        # 3. 连续向量序列的一维位置嵌入
        self.pos_emb = nn.Embedding(args.max_seq_length, args.hidden_size)
        # 4. 堆叠 2 层 AdaLN-Transformer 块
        self.blocks = nn.ModuleList([
            DiTBlock(args.hidden_size, num_heads=4, dropout=0.1) for _ in range(2)
        ])
        self.final_layer = nn.Linear(args.hidden_size, args.hidden_size)

    def forward(self, x_t, t, h_seq, h_sem):
        batch_size, seq_len, _ = x_t.shape
        '''# ================== 🛠️ 就在你的 models.py 物理行插入这段黄金调试 ==================
        if torch.equal(h_seq, h_sem):
            # 🚨 如果触发这个报错，说明你遇到了最隐蔽的“传参乌龙”BUG！
            raise ValueError(
                "🚨【致命传参错误】后台检测到你传给 TrajectoryDiT 的 h_seq 和 h_sem 居然完全相等！\n"
                "这说明你在 trainers.py 调用 dit_model 时，把时序特征（eval_output）同时赋值给了这两个参数。\n"
                "导致所谓的多模态全局语义条件（h_sem）在物理上成了空壳，模型根本没有看到多模态长尾特征！"
            )
        else:
            # 🍏 如果成功打印这个，说明你的 trainers.py 数据流完美闭环，传入了完全独立的不同语义！
            print(f"🍏 [TrajectoryDiT 双条件检查通过] "
                  f"h_seq(时序) 均值: {h_seq.mean().item():.4f} | "
                  f"h_sem(语义) 均值: {h_sem.mean().item():.4f}")'''
        # ==============================================================================
        # 融合时间步特征与 [h_seq, h_sem] 双条件
        t_emb = self.t_embed(t)
        cond_raw = torch.cat([h_seq, h_sem], dim=-1)
        c_emb = self.cond_fusion(cond_raw) + t_emb
        
        # 施加一维位置特征
        positions = torch.arange(seq_len, device=x_t.device).unsqueeze(0).repeat(batch_size, 1)
        x = x_t + self.pos_emb(positions)
        
        # 穿过 DiT 骨干网络
        for block in self.blocks:
            x = block(x, c_emb)
            
        return self.final_layer(x) # 预测返回对应的噪声矩阵
    '''def forward(self, x_t, t, h_seq, h_sem):
        # 🟢 【自适应防御 A】 兼容单步推理时的 2D 输入 [seq_len, hidden_size]
        is_2d_input = False
        if len(x_t.shape) == 2:
            is_2d_input = True
            # 强行升维成标准 3D 方便内部统一计算: [1, seq_len, hidden_size]
            x_t = x_t.unsqueeze(0) 
            
        batch_size, seq_len, _ = x_t.shape
        
        # ================== 🛠️ 就在你的 models.py 物理行插入这段黄金调试 ==================
        if torch.equal(h_seq, h_sem):
            # 🚨 如果触发这个报错，说明你遇到了最隐蔽的“传参乌龙”BUG！
            raise ValueError(
                "🚨【致命传参错误】后台检测到你传给 TrajectoryDiT 的 h_seq 和 h_sem 居然完全相等！\n"
                "这说明你在 trainers.py 调用 dit_model 时，把时序特征（eval_output）同时赋值给了这两个参数。\n"
                "导致所谓的多模态全局语义条件（h_sem）在物理上成了空壳，模型根本没有看到多模态长尾特征！"
            )
        # ==============================================================================
        
        # 🟢 【自适应防御 B】确保时间步 t 的维度与 batch_size 绝对对齐
        if t.ndim == 0:  # 如果是测试集传进来的 0D 标量
            t = t.unsqueeze(0) # 变成 [1]
        t_emb = self.t_embed(t) # 形状: [B, d]
        
        # 🟢 彻底落实专利与论文的核心创新点：双条件拼接引导 C = [h_seq, h_sem]
        # h_seq: [B, d], h_sem: [B, d] -> cond_raw: [B, 2d]
        if h_seq.ndim == 1:
            h_seq = h_seq.unsqueeze(0)
        if h_sem.ndim == 1:
            h_sem = h_sem.unsqueeze(0)
            
        cond_raw = torch.cat([h_seq, h_sem], dim=-1)
        
        # 融合条件特征与时间步特征
        c_emb = self.cond_fusion(cond_raw) + t_emb # [B, d]
        
        # 施加一维位置特征
        positions = torch.arange(seq_len, device=x_t.device).unsqueeze(0).repeat(batch_size, 1)
        x = x_t + self.pos_emb(positions)
        
        # 穿过 DiT 骨干网络
        for block in self.blocks:
            x = block(x, c_emb)
            
        out = self.final_layer(x) # 预测返回对应的噪声矩阵 [B, seq_len, hidden_size]
        
        # 🟢 【自适应防御 C】如果进来的是 2D，出去也降回 2D，完美兼容你的 trainers.py 外挂
        if is_2d_input:
            out = out.squeeze(0)
            
        return out'''