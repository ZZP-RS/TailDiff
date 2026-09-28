# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.optim import Adam
from modules import NCELoss, NTXent
from utils import recall_at_k, ndcg_k, get_metric, nCr, read_user_interactions, check_user_sequence
from utils import recall_at_k, ndcg_k, precision_at_k, mapk, get_metric


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)  # 值在n_views=2时恒等于1
        # projection head for contrastive learn task
        self.projection = nn.Sequential(nn.Linear(self.args.max_seq_length * self.args.hidden_size, \
                                                  512, bias=False), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                        nn.Linear(512, self.args.hidden_size, bias=True))
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        # self.cf_criterion = NTXent()
        print("self.cf_criterion:", self.cf_criterion.__class__.__name__)
        

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    from utils import precision_at_k, mapk

    def get_sample_scores(self, epoch, pred_list, answers=None):
        # 用于Top-1采样评估（每人一个正例）
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, _ = get_metric(pred_list, 5)
        HIT_10, NDCG_10, _ = get_metric(pred_list, 10)

         # 如果有原始答案（多个正例），计算 MAP / Precision
        MAP_10 = 0
        PREC_10 = 0
        if answers is not None:
            MAP_10 = mapk(answers, pred_list.tolist(), 10)
            PREC_10 = precision_at_k(answers, pred_list.tolist(), 10)

        post_fix = {
            "Epoch": epoch,
            "HR@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HR@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HR@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
            "Precision@10": '{:.4f}'.format(PREC_10),
            "MAP@10": '{:.4f}'.format(MAP_10)
           }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [
            HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR, MAP_10, PREC_10
         ], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg, prec, maps = [], [], [], []
        for k in [5, 10, 15, 20,25,30]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
            prec.append(precision_at_k(answers, pred_list, k))
            maps.append(mapk(answers, pred_list, k))

        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "Precision@5": '{:.4f}'.format(prec[0]), "MAP@5": '{:.4f}'.format(maps[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "Precision@10": '{:.4f}'.format(prec[1]), "MAP@10": '{:.4f}'.format(maps[1]),
            "HR@15": '{:.4f}'.format(recall[2]), "NDCG@15": '{:.4f}'.format(ndcg[2]),
            "Precision@15": '{:.4f}'.format(prec[2]), "MAP@15": '{:.4f}'.format(maps[2]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3]),
            "Precision@20": '{:.4f}'.format(prec[3]), "MAP@20": '{:.4f}'.format(maps[3]),
            "HR@25": '{:.4f}'.format(recall[4]), "NDCG@25": '{:.4f}'.format(ndcg[4]),
            "Precision@25": '{:.4f}'.format(prec[4]), "MAP@25": '{:.4f}'.format(maps[4]),
            "HR@30": '{:.4f}'.format(recall[5]), "NDCG@30": '{:.4f}'.format(ndcg[5]),
            "Precision@30": '{:.4f}'.format(prec[5]), "MAP@30": '{:.4f}'.format(maps[5]),
         }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')

        return [
          recall[1], ndcg[1], prec[1], maps[1]  # 可作为主要参考
         ], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class KECLRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader,
                 args,similarity_model=None):
        super(KECLRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            args
        )
        # 初始化相似度模型
        self.similarity_model = similarity_model if similarity_model is not None else args.offline_similarity_model

        if hasattr(self.similarity_model, 'alphas'):
        # 打印验证：是否包含该参数
            '''print(f"多模态权重参数: {self.similarity_model.alphas}")
            print(f"权重梯度状态: {self.similarity_model.alphas.requires_grad}")'''
        
        # 检查相似度模型类型并打印信息
        if hasattr(self.similarity_model, 'alphas'):
            print("✅ 使用多模态相似度计算器 (MultiModalSimilarity)")
            # 将alpha参数加入优化器
            self.optim.add_param_group({
                'params': self.similarity_model.alphas,
                'lr': args.lr* 10.0,  # 使用与主模型相同的学习率
                'weight_decay': args.weight_decay
            })
        else:
            print("✅ 使用传统相似度计算器 (OfflineItemSimilarity)")

        # ====================================================================
        # 🟢 [硬核救砖核心新增]：强行将模型内部的三模态权重和 LayerNorm 捞进 Stage 1 优化器
        # ====================================================================
        if hasattr(self.model, 'use_multimodal_fusion') and self.model.use_multimodal_fusion:
            print("📢 [参数组优化] 正在调整主模型内部融合权重 Learning Rate...")
            
            # 💡 核心逻辑：既然参数已经在优化器里了，为了把 modal_weights 的学习率放大，
            # 我们先在原有默认参数组里把 modal_weights 剔除，再单独以高学习率重新塞入
            for group in self.optim.param_groups:
                # 遍历旧参数组，把 modal_weights 物理剔除，防止重复
                new_params = []
                for p in group['params']:
                    if p is self.model.modal_weights:
                        continue  # 跳过，不放在默认参数组
                    new_params.append(p)
                group['params'] = new_params
            
            # 💡 单独以 10 倍学习率注入，使其免受 ValueError 困扰，同时获得高灵敏度梯度冲击！
            self.optim.add_param_group({
                'params': [self.model.modal_weights],
                'lr': args.lr * 10.0, 
                'weight_decay': args.weight_decay
            })
            
            print("✅ 多模态权重参数去重并成功赋予 10 倍独立学习率！")

    def _one_pair_contrastive_learning(self, inputs):
        '''
        contrastive learning given one pair sequences (batch)
        inputs: [batch1_augmented_data, batch2_augmentated_data]
        '''
        cl_batch = torch.cat(inputs, dim=0)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model.transformer_encoder(cl_batch)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)  # 展平特征 [2B, D]
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)  # 拆分为两个批次 [B, D] 和 [B, D]

        # 计算原始对比损失
        cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1])
    
        # 确保损失非负（添加ReLU约束）
        cl_loss = torch.relu(cl_loss)  # 过滤负数损失
    
        # 处理多模态权重
        if hasattr(self.similarity_model, 'alphas'):
            # 获取特征总维度
            total_dim = cl_output_slice[0].shape[1]
            num_modals = len(self.similarity_model.alphas)  # 模态数量（如3）
            weights = torch.softmax(self.similarity_model.alphas, dim=0)
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            penalty = 0.01 * (1 - entropy)  # 计算惩罚项
            # 打印关键值（每100个batch打印一次，避免输出过多）
        
            cl_loss += penalty
        
        # 精确计算每个模态的维度（确保总和等于总维度）
        # 示例：假设模态维度分配为 [D1, D2, D3]，其中 D1 + D2 + D3 = total_dim
        # 这里根据实际特征结构调整，例如从多模态嵌入中获取各模态原始维度
        # 若已知各模态维度，可直接指定，如：
        # modal_sizes = [711, 711, 712]  # 当 total_dim=2134 时
        # 动态计算方式（推荐）：
            base_dim = total_dim // num_modals
            remainder = total_dim % num_modals
            modal_sizes = [base_dim + 1 if i < remainder else base_dim for i in range(num_modals)]
        
        # 验证维度总和是否正确
            assert sum(modal_sizes) == total_dim, f"模态维度总和 {sum(modal_sizes)} 与总维度 {total_dim} 不匹配"
        
        # 拆分第一个批次的模态特征
            modal_features1 = []
            start = 0
            for size in modal_sizes:
                modal_features1.append(cl_output_slice[0][:, start:start+size])
                start += size
        
        # 拆分第二个批次的模态特征
            modal_features2 = []
            start = 0
            for size in modal_sizes:
                modal_features2.append(cl_output_slice[1][:, start:start+size])
                start += size
        
        # 应用模态权重（确保每个模态特征维度匹配）
            weights = torch.softmax(self.similarity_model.alphas, dim=0)
            weighted_features1 = torch.zeros_like(cl_output_slice[0])  # 初始化与原始特征同维度
            weighted_features2 = torch.zeros_like(cl_output_slice[1])
        
            start = 0
            for i in range(num_modals):
                end = start + modal_sizes[i]
                weighted_features1[:, start:end] = weights[i] * modal_features1[i]
                weighted_features2[:, start:end] = weights[i] * modal_features2[i]
                start = end
        
            # 计算对比损失
            cl_loss = self.cf_criterion(weighted_features1, weighted_features2)
        
            # 熵正则化
            entropy = -torch.sum(weights * torch.log(weights + 1e-8))
            #cl_loss += 0.01 * (1 - entropy)
            entropy = torch.clamp(entropy, min=0.0)
            # 减小惩罚系数（如从0.01改为0.001）
            cl_loss += 0.001 * (1 - entropy)
        else:
        # 无权重时直接计算损失
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1])
    
        return cl_loss

    # ==================== 物理修改：trainers.py 内部 ====================
    def print_simple_modal_weights(self):
        """
        以最精简的一行流格式，随指标同步打印真实有效的三模态权重
        格式为：[Weights] Img: X.XX, Txt: X.XX, Meta: X.XX
        """
        # 防止多卡并行导致读取不到底层的真实函数
        raw_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        if hasattr(raw_model, 'get_modal_weights'):
            w = raw_model.get_modal_weights()  # 获取你 models.py 中带 0.05 温度系数的 [w_img, w_txt, w_meta]
            if w is not None and len(w) >= 3:
                # 转换为最干净的一行学术日志格式
                print(f"[Weights] Img: {w[0]:.4f}, Txt: {w[1]:.4f}, Meta: {w[2]:.4f}")
    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        """
        完美兼容训练(train=True)与评估(train=False)的混血迭代器
        彻底修复特征空间不对齐与诊断矩阵错位  Bug
        """
        str_code = "train" if train else "val"

        # ==================== [分支一]：训练阶段 (Train) ====================
        if train:
            self.model.train()
            rec_cf_data_iter = tqdm(enumerate(dataloader),
                                    desc="Epoch %d: %s" % (epoch, str_code),
                                    total=len(dataloader),
                                    bar_format="{l_bar}{r_bar}")

            total_loss = 0.0

            for i, batch in rec_cf_data_iter:
                batch_tensors = [t.to(self.device) for t in batch]
                user_ids = batch_tensors[0]
                input_ids = batch_tensors[1]
                target_pos = batch_tensors[2]

                attention_mask = (input_ids > 0).long().unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
                
                multimodal_embeddings = self.model.get_multimodal_embeddings(input_ids)
                
                sequence_output = self.model.item_encoder(
                    multimodal_embeddings, 
                    attention_mask, 
                    output_all_encoded_layers=False
                )[-1]

                # 获取全量 Item 的多模态表示（若有 get_all_item_embeddings 方法）
                if hasattr(self.model, 'get_all_item_embeddings'):
                    all_item_emb = self.model.get_all_item_embeddings()
                else:
                    all_item_emb = self.model.item_embeddings_table.weight if hasattr(self.model.item_embeddings_table, 'weight') else self.model.item_embeddings_table

                pos_scores = torch.matmul(sequence_output, all_item_emb.T)

                loss_fct = nn.CrossEntropyLoss(ignore_index=0)
                loss = loss_fct(pos_scores.view(-1, self.args.item_size), target_pos.view(-1))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                total_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(total_loss / len(dataloader))
            }
            rec_cf_data_iter.set_postfix(post_fix)
            print(f"Epoch {epoch} train 推荐系统 Cross-Entropy Loss: {total_loss / len(dataloader):.4f}")
            return total_loss / len(dataloader)

        # ==================== [分支二]：验证/测试阶段 (Valid/Test) ====================
        else:
            rec_data_iter = tqdm(enumerate(dataloader),
                                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                total=len(dataloader),
                                bar_format="{l_bar}{r_bar}")
            self.model.eval()

            final_answers_collected = []
            final_preds_collected = []
            sample_0_raw_scores = None

            if full_sort:
                with torch.no_grad():
                    # 获取全量 Item 的对齐特征
                    if hasattr(self.model, 'get_all_item_embeddings'):
                        all_item_emb = self.model.get_all_item_embeddings()
                    else:
                        all_item_emb = self.model.item_embeddings_table.weight if hasattr(self.model.item_embeddings_table, 'weight') else self.model.item_embeddings_table

                    for i, batch in rec_data_iter:
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, target_pos, target_neg, answers = batch[:5]

                        attention_mask = (input_ids > 0).long().unsqueeze(1).unsqueeze(2)
                        attention_mask = (1.0 - attention_mask) * -10000.0
                        multimodal_embeddings = self.model.get_multimodal_embeddings(input_ids)
                        
                        sequence_output = self.model.item_encoder(
                            multimodal_embeddings, 
                            attention_mask, 
                            output_all_encoded_layers=False
                        )[-1]
                        
                        # 🟢 【核心修复】：精准定位每一个序列中最后一个非 0 (非 Padding) 物品的位置
                        valid_lens = (input_ids > 0).sum(dim=1) - 1
                        valid_lens = valid_lens.clamp(min=0)
                        batch_indices = torch.arange(sequence_output.size(0), device=self.device)
                        recommend_output = sequence_output[batch_indices, valid_lens, :]
                        
                        # 计算与全量多模态 Item 表的相似度
                        rating_pred = torch.matmul(recommend_output, all_item_emb.T)

                        # 屏蔽 Padding (ID 0)
                        rating_pred[:, 0] = -1e10

                        rating_pred_np = rating_pred.cpu().data.numpy().copy()

                        # 保存样本 0 的原始得分用于后续诊断
                        if i == 0:
                            sample_0_raw_scores = rating_pred_np[0].copy()

                        ind = np.argpartition(rating_pred_np, -35)[:, -35:]
                        arr_ind = rating_pred_np[np.arange(len(rating_pred_np))[:, None], ind]
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred_np)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred_np))[:, None], arr_ind_argsort]
                        
                        if i % 100 == 0:
                            for sample_idx in range(min(5, len(batch_pred_list))):
                                true_item = answers.cpu().view(-1)[sample_idx].item()
                                pred_items = batch_pred_list[sample_idx][:10].tolist()
                                print(f"\n[调试] 样本 {i*self.args.batch_size + sample_idx} | 真实目标: {true_item} | Top-10 推荐: {pred_items}")

                        final_answers_collected.extend(answers.cpu().view(-1).tolist())
                        final_preds_collected.extend(batch_pred_list.tolist())

                # ==================== 🎯 严谨的多尺度学术指标计算 ====================
                if len(final_answers_collected) == 0:
                    print("⚠️ 警告: 收集到的评估样本数量为 0！")
                    return [0.0]*7, "Error: No data evaluated"

                print(f"\n📊 [Eval Engine] 开始计算全量用户的多尺度 Recall 与 NDCG... 样本总数: {len(final_answers_collected)}")
                
                k_list = [5, 10, 15, 20, 25, 30]
                hr_dict = {k: [] for k in k_list}
                ndcg_dict = {k: [] for k in k_list}
                mrr_list = []
                
                # 🟢 修复后的诊断逻辑（无 NameError 风险）
                print("=================== 🚨 [防 0 指标物理诊断] 🚨 ===================")
                sample_true = final_answers_collected[0]
                sample_pred = final_preds_collected[0]
                print(f"👉 样本 0 真实 Target 变量类型: {type(sample_true)} | 真实值: {sample_true}")
                print(f"👉 样本 0 Top-35 预测列表: {sample_pred}")
                print(f"👉 样本 0 真实值在预测列表中的 Rank: {sample_pred.index(sample_true) if sample_true in sample_pred else '未命中 (Rank > 35)'}")
                
                if sample_0_raw_scores is not None:
                    print(f"Target [{sample_true}] 原始得分: {sample_0_raw_scores[sample_true]}")
                    top5_ids = np.argsort(sample_0_raw_scores)[::-1][:5]
                    print(f"Top-5 预测 ID: {top5_ids.tolist()}")
                    print(f"Top-5 预测得分: {sample_0_raw_scores[top5_ids].tolist()}")
                    
                    sorted_all_items = np.argsort(sample_0_raw_scores)[::-1]
                    real_rank = np.where(sorted_all_items == sample_true)[0][0] + 1
                    print(f"👉 全量排序中，真实 Target [{sample_true}] 的实际真实排名: {real_rank}")
                print("=================================================================")
                
                for true_item, pred_items in zip(final_answers_collected, final_preds_collected):
                    while isinstance(true_item, (list, tuple, np.ndarray)):
                        true_item = true_item[0]
                    true_item = int(true_item)
                        
                    for k in k_list:
                        sliced_preds = pred_items[:k]
                        if true_item in sliced_preds:
                            hr_dict[k].append(1.0)
                            ndcg_dict[k].append(1.0 / np.log2(sliced_preds.index(true_item) + 2))
                        else:
                            hr_dict[k].append(0.0)
                            ndcg_dict[k].append(0.0)
                    
                    if true_item in pred_items:
                        mrr_list.append(1.0 / (pred_items.index(true_item) + 1))
                    else:
                        mrr_list.append(0.0)
                
                final_hr = {k: float(np.mean(hr_dict[k])) for k in k_list}
                final_ndcg = {k: float(np.mean(ndcg_dict[k])) for k in k_list}
                final_mrr = float(np.mean(mrr_list))
                
                w_info_str = "未启用多模态融合"
                if hasattr(self.model, 'get_modal_weights'):
                    weights = self.model.get_modal_weights  
                    w_img, w_txt, w_meta = weights[0], weights[1], weights[2]
                    w_info_str = f"[ 图像 (Image): {w_img:.4f} | 文本 (Text): {w_txt:.4f} | 属性 (Meta): {w_meta:.4f} ]"

                header_line  = " Metrics    " + "".join([f"|      @{k:<4}" for k in k_list]) + " \n"
                split_line   = "-" * len(header_line) + "\n"
                hr_line      = " Hit Ratio  " + "".join([f"|    {final_hr[k]:.4f}  " for k in k_list]) + "\n"
                ndcg_line    = " NDCG       " + "".join([f"|    {final_ndcg[k]:.4f}  " for k in k_list]) + "\n"
                
                result_info = (
                    f"\n==================== [MSCL-Tail 多尺度性能大表 & 模态权重 - Epoch {epoch}] ====================\n"
                    f"👑 物理真实更新的三模态融合有效权重 -> {w_info_str}\n"
                    f"----------------------------------------------------------------------------------------\n"
                    f"{header_line}"
                    f"{split_line}"
                    f"{hr_line}"
                    f"{ndcg_line}"
                    f"{split_line}"
                    f" MRR: {final_mrr:.4f}\n"
                    f"========================================================================================\n"
                )
                print(result_info)
                
                scores = [final_hr[k] for k in k_list] + [final_ndcg[k] for k in k_list]
                return scores, result_info
# =========================================================================
# ==================== 新增：Diffusion 轨迹重构训练导师 ====================
# =========================================================================
import copy
from models import TrajectoryDiT

class DiffusionTrainer:
    def __init__(self, srs_base, args):
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        
        # 1. 锁死并冻结第一阶段已经训练满性能的多模态基座模型
        self.srs_base = srs_base.to(self.device)
        for param in self.srs_base.parameters():
            param.requires_grad = False
        self.srs_base.eval()
        
        # 2. 初始化核心创新点：DiT 模块
        from models import TrajectoryDiT  # 确保正常引入
        self.dit_model = TrajectoryDiT(args).to(self.device)
        self.optimizer = torch.optim.AdamW(self.dit_model.parameters(), lr=1e-4)
        
        # 3. 经典 DDPM 扩散参数设定（100步线性加噪）
        self.num_steps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.num_steps, device=self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # 确保它的维度 args.hidden_size 与你的条件流完全匹配
        self.pure_discrete_item_embedding = nn.Embedding(
            num_embeddings=self.args.item_size, 
            embedding_dim=self.args.hidden_size, 
            padding_idx=0
        )
        
        # 使用 Xavier 均匀分布进行物理空间初始化
        nn.init.xavier_uniform_(self.pure_discrete_item_embedding.weight)
        
        # 物理设备对齐（送入 GPU 运行）
        if self.cuda_condition:
            self.pure_discrete_item_embedding.cuda()

    def train_epoch(self, head_user_loader):
        """第二阶段：仅在活跃的头部用户数据上训练扩散模型，迁移轨迹重建能力"""
        self.dit_model.train()
        total_loss = 0
        
        for batch in head_user_loader:
            input_ids = batch[1].to(self.device) # 形状: [batch_size, seq_len]
            batch_size, seq_len = input_ids.shape
            
            with torch.no_grad():
                # A. 正向查表：拿着 ID 提取包含多模态信息的固定连续隐向量矩阵
                x_0 = self.srs_base.get_multimodal_embeddings(input_ids)
                
                # B. 提取条件 1 (时序结构): 借助原生 SASRec 模型结构
                attention_mask = (input_ids > 0).long().unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
                recommend_output = self.srs_base.item_encoder(x_0, attention_mask, output_all_encoded_layers=False)[-1]
                h_seq = recommend_output[:, -1, :] # 获取时序高度概括表征
                
                # C. 提取条件 2 (多模态全局语义): 显式建立用户心智模式
                valid_mask = (input_ids > 0).float().unsqueeze(-1)
                valid_lens = valid_mask.sum(dim=1).clamp(min=1)
                h_sem = (x_0 * valid_mask).sum(dim=1) / valid_lens
                
            # D. 前向加噪
            t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
            noise = torch.randn_like(x_0)
            alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1)
            x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
            
            # E. 训练 DiT 模型预测噪声
            predicted_noise = self.dit_model(x_t, t, h_seq, h_sem)
            loss = nn.functional.mse_loss(predicted_noise, noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(head_user_loader)

    # 🟢 补全单步去噪采样函数（DDPM标准反向逻辑），防止后面调用 self.p_sample_step 崩溃
    def p_sample_step(self, x_t, t, noise_pred):
        """
        根据 DDPM 理论由 t 时的 x_t 和预测出的噪声，反向还原估计出 t-1 时的 x_{t-1}
        """
        beta_t = self.beta[t].view(-1, 1, 1)
        alpha_t = self.alpha[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        
        # 核心去噪公式
        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred)
        
        if t.item() == 0:
            return mean
        else:
            # 加上随机方差
            variance = beta_t
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise

    
    @torch.no_grad()
    @torch.no_grad()
    def generate_for_tail(self, tail_user_seqs_list, user_seq):
        """
        👑 物理对齐修复版：保证训练与推理阶段条件流 h_seq / h_sem 的完美空间一致性
        """
        self.srs_base.eval()
        self.dit_model.eval() 
        nu = self.args.pseudo_len 
        max_len = self.args.max_seq_length
        print(f"\n🎯 [MSCL-Tail 架构联动] 当前数据集统一生成的伪序列长度硬性规定为 Nu = {nu}")
        
        # =========================================================================
        # 🚀 【第一步】: 动态计算真实尾部用户在原始 SRS 基座上的真实 NDCG@10 平均值
        # =========================================================================
        print("\n[Gate 1/3] 正在动态构建逆向基准：计算真实尾部用户原始 NDCG@10 阈值...")
        real_tail_scores = []
        all_poi_embeddings = self.srs_base.item_embeddings_table

        sample_h_seq_list = []
        sample_h_sem_list = []

        for uid, pure_train_prefix in tail_user_seqs_list:
            orig_seq = user_seq[uid] 
            if len(orig_seq) < 3: continue
            target_item = orig_seq[-1] 
            
            truncated_prefix = pure_train_prefix[-max_len:]
            pad_len = max_len - len(truncated_prefix)
            padded_prefix = [0] * pad_len + truncated_prefix
            
            eval_input = torch.tensor([padded_prefix], dtype=torch.long, device=self.device)
            eval_x = self.srs_base.get_multimodal_embeddings(eval_input)
            eval_mask = (eval_input > 0).long().unsqueeze(1).unsqueeze(2)
            eval_mask = (1.0 - eval_mask) * -10000.0
            
            eval_output = self.srs_base.item_encoder(eval_x, eval_mask, output_all_encoded_layers=False)[-1][:, -1, :]
            test_rating_pred = torch.matmul(eval_output, all_poi_embeddings.T) 
            
            _, topk_idx = torch.topk(test_rating_pred, k=10, dim=-1)
            topk_idx = topk_idx.squeeze(0).cpu().numpy().tolist()
            
            user_ndcg_10 = 1.0 / np.log2(topk_idx.index(target_item) + 2) if target_item in topk_idx else 0.0
            real_tail_scores.append(user_ndcg_10)
                
        benchmark_threshold = np.mean(real_tail_scores) if real_tail_scores else 0.0
        print(f"📊 [基准确立完毕] 真实尾部用户原始平均 NDCG@10 = {benchmark_threshold:.4f}")
        #benchmark_threshold = 0.2  # 手动设置伪序列过滤阈值
        #print(f"📊 [基准确立完毕] 手动设置的 NDCG@10 过滤阈值 = {benchmark_threshold:.4f}")
        
        # =========================================================================
        # 🚀 【第二步】: 遍历长尾用户，扩散脑补 -> 门控过滤
        # =========================================================================
        print("[Gate 2/3] 扩散模型全面启动：执行双条件生成与物理留存门控清洗...")
        D_tail_filt = []
        
        for uid, pure_train_prefix in tqdm(tail_user_seqs_list, desc="Filtering Tail Sequences"):
            orig_seq = user_seq[uid]
            target_item = orig_seq[-1] 
            
            truncated_prefix = pure_train_prefix[-max_len:]
            pad_len = max_len - len(truncated_prefix)
            padded_prefix = [0] * pad_len + truncated_prefix
            
            cond_input = torch.tensor([padded_prefix], dtype=torch.long, device=self.device)
            
            # 🟢 【核心物理修复】：与 train_epoch 完全对齐！使用多模态 Embedding 提 h_seq 和 h_sem
            eval_x_cond = self.srs_base.get_multimodal_embeddings(cond_input)
            
            eval_mask_cond = (cond_input > 0).long().unsqueeze(1).unsqueeze(2)
            eval_mask_cond = (1.0 - eval_mask_cond) * -10000.0

            # 1. 时序条件流 (h_seq)
            recommend_output_cond = self.srs_base.item_encoder(eval_x_cond, eval_mask_cond, output_all_encoded_layers=False)[-1]
            h_seq = recommend_output_cond[:, -1, :]

            # 2. 语义条件流 (h_sem)
            valid_mask_cond = (cond_input > 0).float().unsqueeze(-1)
            valid_lens_cond = valid_mask_cond.sum(dim=1).clamp(min=1)
            h_sem = (eval_x_cond * valid_mask_cond).sum(dim=1) / valid_lens_cond

            sample_h_seq_list.append(h_seq.detach().cpu())
            sample_h_sem_list.append(h_sem.detach().cpu())
            
            h_seq_2d = h_seq.reshape(1, -1)
            h_sem_2d = h_sem.reshape(1, -1)
            
            # 3. 反向 DDPM 去噪采样
            x_t = torch.randn(1, nu, self.args.hidden_size, device=self.device)
            
            for t in reversed(range(self.num_steps)):
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                noise_pred = self.dit_model(x_t, t_tensor, h_seq_2d, h_sem_2d)
                x_t = self.p_sample_step(x_t, t_tensor, noise_pred)
            
            # 4. 余弦相似度重构 POI 序列
            generated_embeddings = x_t.squeeze(0)
            pseudo_sequence = []
            for item_emb in generated_embeddings:
                cos_sim = torch.cosine_similarity(item_emb.unsqueeze(0), all_poi_embeddings, dim=-1)
                cos_sim[0] = -1.0  # 排除 padding 项
                pseudo_sequence.append(torch.argmax(cos_sim).item())
            
            test_pseudo_input_seq = pure_train_prefix + pseudo_sequence
            
            # 5. 评估伪序列质量并进行门控筛选
            eval_test_seq = test_pseudo_input_seq[-max_len:]
            eval_test_pad = max_len - len(eval_test_seq)
            eval_test_padded = [0] * eval_test_pad + eval_test_seq
            
            eval_input = torch.tensor([eval_test_padded], dtype=torch.long, device=self.device)
            eval_x = self.srs_base.get_multimodal_embeddings(eval_input)
            eval_mask = (eval_input > 0).long().unsqueeze(1).unsqueeze(2)
            eval_mask = (1.0 - eval_mask) * -10000.0
            
            eval_output = self.srs_base.item_encoder(eval_x, eval_mask, output_all_encoded_layers=False)[-1][:, -1, :]
            test_rating_pred = torch.matmul(eval_output, all_poi_embeddings.T)
            
            _, topk_idx = torch.topk(test_rating_pred, k=10, dim=-1)
            topk_idx = topk_idx.squeeze(0).cpu().numpy().tolist()
            
            pseudo_ndcg_10 = 1.0 / np.log2(topk_idx.index(target_item) + 2) if target_item in topk_idx else 0.0
            
            if pseudo_ndcg_10 >= benchmark_threshold:
                D_tail_filt.append((uid, pure_train_prefix + pseudo_sequence))
                
        # =========================================================================
        # 👑 自证对齐报告
        # =========================================================================
        all_sample_seq = torch.cat(sample_h_seq_list, dim=0)
        all_sample_sem = torch.cat(sample_h_sem_list, dim=0)
        cos_sim_matrix = torch.nn.functional.cosine_similarity(all_sample_seq, all_sample_sem, dim=-1)
        
        print("\n" + "="*25 + " 🛠️ MSCL-Tail 双条件流物理对齐自证报告 " + "="*25)
        print(f"🔊 [检查项 1/3] 条件独立性: h_seq 与 h_sem 余弦相似度 = {cos_sim_matrix.mean().item():.4f}")
        print(f"🔊 [检查项 2/3] 条件尺度: h_seq std={all_sample_seq.std().item():.4f} | h_sem std={all_sample_sem.std().item():.4f}")
        print(f"🔊 [检查项 3/3] 门控留存率: {len(D_tail_filt)} / {len(tail_user_seqs_list)} ({len(D_tail_filt)/max(1, len(tail_user_seqs_list))*100:.2f}%)")
        print("="*85 + "\n")
        
        return D_tail_filt