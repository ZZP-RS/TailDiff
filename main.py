# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch
import argparse
from datetime import datetime, timedelta
import logging
import json
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from datasets import RecWithContrastiveLearningDataset
from models import OfflineItemSimilarity, SASRecModel
from trainers import KECLRecTrainer
from utils import check_path, set_seed, EarlyStopping, get_threshold_value
from utils import get_user_seqs_rich, generate_soft_mask_matrix_valid, generate_soft_mask_matrix_test
from models import MultiModalSimilarity  # 假设您把新类放在models.py中
import sys
import io
import copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        value = getattr(args, arg)
        if value is None:
            value = "None"  # 将 None 转为字符串，避免格式化错误
        if isinstance(value, list):
            value_str = str(value)
            print(f"{arg:<30} : {value_str:>35}")
        else:
            print(f"{arg:<30} : {value:>35}")

def identify_long_tail_users_by_user_ratio(user_seq, ratio=0.2):
    user_lengths = [(i, len(seq)) for i, seq in enumerate(user_seq)]
    user_lengths.sort(key=lambda x: x[1])  # 按序列长度升序
    total_interactions = sum([l for _, l in user_lengths])
    cum_sum = 0
    long_tail_user_ids = []
    for uid, length in user_lengths:
        cum_sum += length
        long_tail_user_ids.append(uid)
        if cum_sum >= total_interactions * ratio:
            break
    return set(long_tail_user_ids)

    # 全体训练、长尾测试（长尾用户有 train/valid/test；非长尾用户只有 train）

#全体训练长尾测试
'''
def split_user_sequences(user_seq, long_tail_users):
    train_seq, valid_seq, test_seq = [], [], []
    for uid, seq in enumerate(user_seq):
        if uid in long_tail_users and len(seq) >= 3:
            train_seq.append(seq[:-2])
            valid_seq.append(seq[:-1])
            test_seq.append(seq)
        else:
            train_seq.append(seq)
            valid_seq.append([])  # 非长尾用户无验证
            test_seq.append([])  # 非长尾用户无测试
    return train_seq, valid_seq, test_seq
#全体测试
'''


'''def split_user_sequences(user_seq, long_tail_users):
    train_seq, valid_seq, test_seq = [], [], []
    for uid, seq in enumerate(user_seq):
        
            train_seq.append(seq[:-2])
            valid_seq.append(seq[:-1])
            test_seq.append(seq)
       
    return train_seq, valid_seq, test_seq'''


# 只保留长尾用户的训练/测试序列,长尾训练长尾测试
def split_user_sequences(user_seq, long_tail_users):
    train_seq, valid_seq, test_seq = [], [], []
    for uid, seq in enumerate(user_seq):
        # 只处理长尾用户，且序列长度至少为3（确保能分割出train/valid/test）
        if uid in long_tail_users and len(seq) >= 3:
            train_seq.append(seq[:-2])  # 训练集：取到倒数第2个元素
            valid_seq.append(seq[:-1])  # 验证集：取到倒数第1个元素
            test_seq.append(seq)        # 测试集：完整序列
        else:
            # 非长尾用户不参与训练和测试，用空序列表示
            train_seq.append([])
            valid_seq.append([])
            test_seq.append([])
    return train_seq, valid_seq, test_seq
def main():
    parser = argparse.ArgumentParser()
    # system args
    parser.add_argument('--data_dir', default='/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='new_orleans', type=str)
    parser.add_argument('--do_eval', action='store_true')
    #0是长长 1是全长 6消融
    parser.add_argument('--model_idx', default=7, type=int, help="model idenfier 10, 20, 30...")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument('--tail_test', action='store_true')
    parser.add_argument('--random_augment_combination', default='IS', type=str,
                        help='SM SR IR... choose any two operators from them to combine.')
    parser.add_argument('--similarity_model_name', default='cosine_similarity', type=str,
                        help="Method to generate item similarity score. choices: \
                        cosine_similarity, HNSW, Random")
    parser.add_argument('--similarity_model_path', type=str, default=None,
                    help='路径用于保存/加载传统相似度模型（如ItemCF、cosine_similarity）')
    parser.add_argument('--base_augment_type', default='insert', type=str,
                        help="default data augmentation types. Chosen from: \
                        mask, crop, reorder, substitute, insert, random, \
                        combinatorial_enumerate (for multi-view).")
    parser.add_argument("--tao", type=float, default=0.1, help="crop ratio for crop operator")
    parser.add_argument("--gamma", type=float, default=0.1, help="mask ratio for mask operator")
    parser.add_argument("--beta", type=float, default=0.1, help="reorder ratio for reorder operator")
    parser.add_argument("--substitute_rate", type=float, default=0.1,
                        help="substitute ratio for substitute operator")
    parser.add_argument("--insert_rate", type=float, default=0.3,
                        help="insert ratio for insert operator")
    parser.add_argument("--max_insert_num_per_pos", type=int, default=1,
                        help="maximum insert items per position for insert operator - not studied")
    parser.add_argument('--use_multimodal', action='store_true', help='是否使用多模态相似度')
    parser.add_argument('--image_sim_path', type=str, help='图像相似度矩阵路径')
    parser.add_argument('--text_sim_path', type=str, help='文本相似度矩阵路径')
    parser.add_argument('--meta_sim_path', type=str, help='元数据相似度矩阵路径')
    parser.add_argument('--poi2id_path', type=str, help='POI到ID映射文件路径') 
    parser.add_argument('--enabled_modals', nargs='+', default=[ 'text','meta','image'],
                        help="启用的模态，可选值: image, text, meta")     
              
              

    # contrastive learning task args
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='softmax temperature (default:  1.0) - not studied.')
    parser.add_argument('--n_views', default=2, type=int, metavar='N',
                        help='Number of augmented data for each sequence - not studied.')

    # model args
    parser.add_argument("--model_name", default='KECLSR', type=str)
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--gen_len', type=int, default=20, help='扩散模型为短序列额外生成的伪节点/伪序列长度')
    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=600, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--cf_weight", type=float, default=0.2, \
                        help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, \
                        help="weight of contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    # ================= 👑 新增：三模态特征独立静态路径参数 👑 =================
    parser.add_argument('--img_embedding_path', default='/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/image_emb.npy', type=str, 
                        help='Path to the pre-trained aligned 768-dim image embedding (.npy file)')
    parser.add_argument('--txt_embedding_path', default='/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/review_emb.npy', type=str, 
                        help='Path to the pre-trained aligned 768-dim text embedding (.npy file)')
    parser.add_argument('--meta_embedding_path', default='/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/meta_emb.npy', type=str, 
                        help='Path to the pre-trained aligned 768-dim metadata embedding (.npy file)')
    # =====================================================================
    args = parser.parse_args()
    args.multimodal_embedding_path = "/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/fused_embedding_pca128.npy"
    args.image_sim_path = "/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/similarity_graphs/image_sim.npy"
    args.text_sim_path = "/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/similarity_graphs/review_sim.npy"
    args.meta_sim_path = "/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/similarity_graphs/meta_sim.npy"
    args.poi2id_path = "/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/poi2id.json"
    args.use_multimodal = True  # 设置为True以启用多模态
    set_seed(args.seed)
    check_path(args.output_dir)  # 检查output路径是否存在,不存在则创建
    
    #
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    args.data_file = args.data_dir + args.data_name + '.txt'
    args.tail_test = False
    args.test_mode = 'tail'



    rich_json_path = "/home/qingjiao/wmy/home/qingjiao/wmy/MSCL-Tail/POI_data/new_orleans/new_orleans_user_rich_seq.json"
    rich_seqs = get_user_seqs_rich(rich_json_path)
    num_users = len(rich_seqs) 

    # ==================== 1. 检查缺失的 POI ID 与映射反查 ====================
    with open(args.poi2id_path, 'r') as f:
        poi2id = json.load(f)
    id2poi = {int(v): k for k, v in poi2id.items()}  # 内部数字ID → 原始字符串ID

    all_poi_in_seq = set()
    user_seq = [[v["item"] for v in user] for user in rich_seqs]  # 提取纯交互 POI ID 序列
    for seq in user_seq:
        for num_id in seq:
            num_id_int = int(num_id)  # 序列中的 ID 是数字（如 2965）
            raw_poi_id = id2poi.get(num_id_int, None)
            if raw_poi_id:
                all_poi_in_seq.add(raw_poi_id)  # 成功反查，加入原始字符串 ID
            else:
                all_poi_in_seq.add(str(num_id_int))  # 找不到则保留数字字符串

    # 对比 poi2id 的 key（原始字符串 ID），找出真正缺失的 POI
    poi_in_mapping = set(poi2id.keys())
    missing_pois = all_poi_in_seq - poi_in_mapping
    if missing_pois:
        print(f"真正缺失的原始 POI ID 数量：{len(missing_pois)}")
        print("前10个缺失的原始 POI ID：", list(missing_pois)[:10])
    else:
        print("所有 POI ID 均已匹配，无缺失！")

    # ==================== 2. 多模态嵌入尺寸配置 ====================
    multimodal_embed = np.load(args.multimodal_embedding_path)
    args.item_size = multimodal_embed.shape[0]  # 3815，与嵌入维度保持一致
    max_item_in_embed = args.item_size - 1      # 嵌入的最大物品 ID 为 3814
    args.mask_id = args.item_size  

    valid_rating_matrix = generate_soft_mask_matrix_valid(rich_seqs, num_users, args.item_size, args.data_name)
    test_rating_matrix = generate_soft_mask_matrix_test(rich_seqs, num_users, args.item_size, args.data_name)

    # ==================== 3. MSCL-Tail 核心：设置生成拼接长度 gen_len ====================
    # 1. 优先使用命令# 1. 优先使用命令行传入的 gen_len，如果没传则默认使用预设值（如 5）
    if not hasattr(args, 'gen_len') or args.gen_len is None:
        args.gen_len = 5  # 👈 这里可以修改你的默认生成拼接长度，比如 2, 5, 8 等

    # 2. 🟢【关键修复】：补上 pseudo_len 参数，防止 trainers.py (line 642) 报 AttributeError
    # 将其直接设为 max_seq_length，确保扩散模型内部调用 self.args.pseudo_len 时不会报错且不过界
    args.pseudo_len = args.max_seq_length

    # 3. 保留日志打印
    all_lengths = [len(seq) for seq in user_seq if len(seq) > 0]
    print(f"📊 [MSCL-Tail] 全局原始序列统计: 平均长度={np.mean(all_lengths):.2f}, 最大长度={np.max(all_lengths)}")
    print(f"🎯 [参数分析实验] 设定扩散模型单次额外拼接长度 (gen_len) = {args.gen_len}")
    print(f"🛡️ [安全限制] 序列总长度限制 (max_seq_length) = {args.max_seq_length}")

   
   
    # ==================== 4. 划分头尾数据集 ====================
    long_tail_users = identify_long_tail_users_by_user_ratio(user_seq, ratio=0.2)
    train_seq, valid_seq, test_seq = split_user_sequences(user_seq, long_tail_users)
    print("用户总数:", len(user_seq))
    print("长尾用户数:", len(long_tail_users))

    # 检查划分后每个集合中“有效序列”的用户数量
    train_valid = [s for s in train_seq if len(s) >= 1]
    valid_valid = [s for s in valid_seq if len(s) >= 2]
    test_valid  = [s for s in test_seq  if len(s) >= 3]

    print("训练集中有效序列数:", len(train_valid))
    print("验证集中有效序列数:", len(valid_valid))
    print("测试集中有效序列数:", len(test_valid))
    
    max_item_in_seq = max([max(seq) for seq in user_seq if len(seq) > 0])
    print("user_seq 中最大 POI id:", max_item_in_seq)

    # 保存模型相关参数
    args_str = f'{args.model_name}-{args.data_name}-{args.model_idx}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    # 保存 Nu 相关信息到日志
     # 日志记录
    with open(args.log_file, 'a', encoding='utf-8') as f:
        f.write(f"[Experiment Config] 额外生成拼接长度 (gen_len): {args.gen_len}, 最大允许序列长度: {args.max_seq_length}\n")
    show_args_info(args)

    with open(args.log_file, 'a', encoding='utf-8') as f:
        f.write(str(args) + '\n')
    print(f"识别出的长尾用户数量：{len(long_tail_users)} / {len(user_seq)}")
    with open(args.log_file, 'a', encoding='utf-8') as f:
        f.write(f"识别出的长尾用户数量：{len(long_tail_users)} / {len(user_seq)}\n")
    
    args.train_matrix = valid_rating_matrix
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # ==================== 5. 相似度计算器初始化与预计算 ====================
    args.similarity_model_path = os.path.join(args.data_dir, args.data_name + '_' + args.similarity_model_name + '_similarity.pkl')

    if args.use_multimodal:
        similarity_model = MultiModalSimilarity(
            image_sim_path=args.image_sim_path,
            text_sim_path=args.text_sim_path,
            meta_sim_path=args.meta_sim_path,
            poi2id_path=args.poi2id_path,
            device="cuda" if args.cuda_condition else "cpu",
            enabled_modals=args.enabled_modals
        )
        print(f"使用多模态相似度计算器，启用模态: {args.enabled_modals}")
    else:
        print("使用传统相似度计算器")
        similarity_model = OfflineItemSimilarity(
            args, 
            data_file=args.data_file,
            similarity_path=args.similarity_model_path,
            model_name=args.similarity_model_name,
            dataset_name=args.data_name
        )
    similarity_model.precompute_topk(top_k=50) # 执行 HNSW 索引预计算
    args.similarity_model = similarity_model
    args.offline_similarity_model = similarity_model
    
    # 构建初始数据集与 Dataloader
    train_dataset = RecWithContrastiveLearningDataset(args, train_seq, data_type='train')
    eval_dataset = RecWithContrastiveLearningDataset(args, valid_seq, data_type='valid', long_tail_user_set=long_tail_users)
    test_dataset = RecWithContrastiveLearningDataset(args, test_seq, data_type='test', long_tail_user_set=long_tail_users)
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Valid Dataset size: {len(eval_dataset)}")
    print(f"Test Dataset  size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=args.batch_size)
    test_dataloader  = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),  batch_size=args.batch_size)

    # ==================== 6. 初始化核心推荐基座模型与 Trainer ====================
    model = SASRecModel(args=args)
    args.model = model
    trainer = KECLRecTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args=args, similarity_model=similarity_model)
    
    # 🍏 物理修复：提前绑定长尾用户集合与原始序列，防止 do_eval 阶段发生 NameError
    trainer.long_tail_user_set = long_tail_users
    trainer.original_user_seq = user_seq

    if args.do_eval:
        trainer.args.train_matrix = test_rating_matrix
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info = trainer.test(0, full_sort=True)
        print(result_info)

    else:
        # ------------ 【Stage 1】: 预训练包含长尾用户的多模态推荐基座 ------------
        print("====== [Stage 1] 开始在原始全量数据集上预训练多模态 SRS 基座 ======")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=100, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)  # 调用原生多模态联合对比学习训练
            scores, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("基座模型预训练收敛！")
                break
                
        # 重新加载 Stage 1 中在全量数据上表现最好的基座权重
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        
        # ------------ 【Stage 2】: 提取头尾用户，启动外挂式一维 DiT 闭门练兵 ------------
        print("====== [Stage 2] 正在划分头部活跃用户与长尾短序列用户 ======")
        long_tail_user_set = identify_long_tail_users_by_user_ratio(user_seq, ratio=0.2)
        
        # 通过集合运算，优雅地得到头部活跃用户集
        head_user_set = set(range(len(user_seq))) - long_tail_user_set
        
        # 仅抽取头部用户的交互序列，用于扩散模型重构知识的学习（无长尾噪声污染）
        head_user_seqs = [seq for uid, seq in enumerate(user_seq) if uid in head_user_set]
        head_dataset = RecWithContrastiveLearningDataset(args, head_user_seqs, data_type='train')
        head_dataloader = DataLoader(head_dataset, sampler=RandomSampler(head_dataset), batch_size=args.batch_size)
        
        # 初始化外挂式数据增强训练扩散导师
        from trainers import DiffusionTrainer
        diff_trainer = DiffusionTrainer(srs_base=trainer.model, args=args)
        
        print("====== [Stage 2] 扩散模型启动：开始仅在头部用户数据上学习轨迹重构知识 ======")
        for diff_epoch in range(200):  # 充分训练扩散模型
            loss = diff_trainer.train_epoch(head_dataloader)
            if (diff_epoch + 1) % 10 == 0:
                print(f"Diffusion Epoch {diff_epoch+1} | 噪声预测 MSE Loss: {loss:.4f}")
            
        # ------------ 【Stage 3 & 4】: 长尾条件映射、全局重构与基座模型过滤清洗 ------------
        print("====== [Stage 3 & 4] 长尾短序列长驱直入：执行双条件全局潜在轨迹恢复 ======")
        
        # 剥离验证集和测试集，只把纯训练前缀喂给扩散模型（切掉后2个考题物品）
        tail_user_seqs_list = []
        for uid, seq in enumerate(user_seq):
            if uid in long_tail_user_set:
                pure_train_prefix = seq[:-2] if len(seq) > 2 else seq
                # 💡 核心逻辑：计算当前用户最多还能拼接多少个节点，防止超过 max_seq_length
                actual_gen_len = min(args.gen_len, args.max_seq_length - len(pure_train_prefix))
        
                if actual_gen_len > 0:
                    tail_user_seqs_list.append((uid, pure_train_prefix))

        # 扩散模型调用：传入指定的 actual_gen_len 长度生成伪序列，生成后拼接在 pure_train_prefix 后面
        D_tail_filt = diff_trainer.generate_for_tail(tail_user_seqs_list, user_seq=user_seq)
        print(f"✅ 数据大丰收！共有 {len(D_tail_filt)} 条长尾增强序列完美通过质量过滤验证！")

        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write("\n================ [Stage 3 & 4] 长尾伪序列过滤结果 ================\n")
            f.write(f"过滤前长尾用户序列数: {len(tail_user_seqs_list)}\n")
            f.write(f"过滤后保留的增强序列数: {len(D_tail_filt)}\n")
            f.write(f"伪序列留存率: {len(D_tail_filt) / max(1, len(tail_user_seqs_list)) * 100:.2f}%\n")
        
        # =========================================================================
        # 👑 【联合微调池构建】：全量真实用户训练集 + 门控高质量伪序列
        # =========================================================================
        test_user_seq = copy.deepcopy(user_seq)  # 锁死一份绝对纯净的原始全量序列备份，供大考使用
        final_train_seq_list = []
        
        # 1. 物理提取所有真实用户的纯训练前缀部分全量入池
        for uid, orig_items in enumerate(test_user_seq):
            pure_orig_train = orig_items[:-2] if len(orig_items) > 2 else orig_items
            final_train_seq_list.append(pure_orig_train)
            
        # 2. 将通过了门控清洗的高质量伪增强序列作为独立的虚拟样本追加进池中
        for uid, pseudo_sequence in D_tail_filt:
            final_train_seq_list.append(pseudo_sequence)
            
        print(f"\n📊 [真正全量池构建完毕] 全量真实用户训练前缀: {len(test_user_seq)} 条")
        print(f"📊 [真正全量池构建完毕] 成功追击的高质量虚拟样本: {len(D_tail_filt)} 条")
        print(f"📊 [真正全量池构建完毕] Stage 5 全量端到端微调总样本数: {len(final_train_seq_list)} 条")
        
        # 构建包含全量+虚拟样本的端到端微调 DataLoader
        final_train_dataset = RecWithContrastiveLearningDataset(args, final_train_seq_list, data_type='pure_train')
        trainer.train_dataloader = DataLoader(final_train_dataset, sampler=RandomSampler(final_train_dataset), batch_size=args.batch_size)

        # ------------ 【Stage 5】: 解冻多模态 Embedding 空间，全面交叉熵最终微调 ------------
        print("====== [Stage 5] 彻底解冻多模态基座 Embedding 与融合权重，全面滋养微调 ======")
        
        # 🍏 物理修复：将解冻逻辑从 epoch 内部移出，只在微调前显式解冻一次，避免每轮循环内的冗余操作
        if hasattr(trainer.model, 'use_multimodal_fusion') and trainer.model.use_multimodal_fusion:
            print("[Info] 正在解冻三模态投影降维层与可学习表示层融合参数...")
            for param in trainer.model.img_proj.parameters(): param.requires_grad = True
            for param in trainer.model.txt_proj.parameters(): param.requires_grad = True
            for param in trainer.model.meta_proj.parameters(): param.requires_grad = True
            trainer.model.modal_weights.requires_grad = True
        else:
            trainer.model.item_embeddings.weight.requires_grad = True

        # 🟢 通过 PyTorch 参数组将常规推荐网络层和模态权重剥离，给予差异化超高学习率！
        if hasattr(trainer.model, 'modal_weights') and trainer.model.modal_weights.requires_grad:
            print("🚀 [优化器重构] 正在为 modal_weights 专门建立超高学习率独立通道...")
            base_params = [
                p for n, p in trainer.model.named_parameters() 
                if p.requires_grad and 'modal_weights' not in n
            ]
            finetune_param_groups = [
                {
                    'params': base_params, 
                    'lr': 1e-4, 
                    'weight_decay': args.weight_decay
                },
                {
                    'params': [trainer.model.modal_weights], 
                    'lr': 1e-2,  # 💡 致命武器：给模态权重放大 100 倍学习率(0.01)，使其快速冲出 Softmax 饱和区！
                    'weight_decay': 0.0  # 标量融合权重不加正则化约束，释放自由度
                }
            ]
            print(f"🔥 常规微调学习率: 1e-4 | 🟢 模态权重专属冲刺学习率: 1e-2 (已放大 100 倍)")
        else:
            finetune_param_groups = [{'params': filter(lambda p: p.requires_grad, trainer.model.parameters()), 'lr': 1e-4, 'weight_decay': args.weight_decay}]

        # 激活最终微调的总梯度通道
        trainer.optim = torch.optim.Adam(finetune_param_groups)
        TOTAL_FINE_TUNE_EPOCHS = max(500, args.epochs)  
        
        # 独立早停：监控最优微调权重另存为 {原权重名}_ft.pt，防止覆盖预训练基座
        fine_tune_early_stopping = EarlyStopping(
            args.checkpoint_path.replace('.pt', '_ft.pt'), 
            patience=30, 
            verbose=True
        )
        
        for final_epoch in range(TOTAL_FINE_TUNE_EPOCHS): 
            # 跑一轮训练，物理同步输出 "推荐系统 Cross-Entropy Loss"
            trainer.train(final_epoch) 
            
            # 严格控制：每隔 10 轮执行一次学术指标测试并判定早停
            if final_epoch == 0 or (final_epoch + 1) % 10 == 0:
                print(f"\n📊 [Stage 5 阶段性微调评测] Epoch {final_epoch + 1} 正在评估验证/测试集...")
                trainer.args.train_matrix = test_rating_matrix
                
                # 调用评测引擎，返回双线大考指标
                scores, result_info = trainer.test(final_epoch, full_sort=True)
                print(result_info)
                
                with open(args.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"--- [Stage 5 微调] Epoch {final_epoch + 1} Mid-Eval ---\n{result_info}\n")
                
                # 早停判定（监控列表倒数第二个元素，即 NDCG@20）
                fine_tune_early_stopping(np.array(scores[-2:-1]), trainer.model)
                if fine_tune_early_stopping.early_stop:
                    print(f"🚀 [Stage 5 触发早停] 模型在第 {final_epoch + 1} 轮已完全收敛，掐断微调阶段。")
                    break
                    
        # ==================== 7. MSCL-Tail 最终大考：模式控制中心 ====================
        print('\n--------------- 👑 [微调全部结束] 正在加载 Stage 5 最优微调权重进行最终离线评测 -------------------')
        try:
            trainer.model.load_state_dict(torch.load(args.checkpoint_path.replace('.pt', '_ft.pt')))
            print("✅ 成功加载微调阶段表现最好的模型权重！")
        except Exception as e:
            print(f"⚠️ 未找到微调最佳权重备份，将直接使用当前最后一轮的权重执行评测。Err: {e}")
        
        # 强制对齐最终测试集的掩码矩阵
        trainer.args.train_matrix = test_rating_matrix
        
        # 🟢 核心物理修复：在传入 Dataset 之前，在主控侧通过真实 uid 矩阵完成精准长尾过滤
        if args.test_mode == 'tail':
            print(f"🎯 [模式锁死] 当前测试模式为: 【仅对长尾用户做测试 (All-Train, Tail-Test)】")
            
            # 严格根据真实的全局用户 ID (uid) 进行筛选，防止 Dataset 内部索引错位
            filtered_test_seq = []
            for uid, seq in enumerate(test_user_seq):
                if uid in long_tail_user_set and len(seq) >= 3:
                    filtered_test_seq.append(seq)
                    
            print(f"📊 [物理过滤完毕] 预期长尾测试有效序列数: {len(filtered_test_seq)}")
            
            # 物理清洗后的序列送入 Dataset，此时 long_tail_user_set 传 None，避免 Dataset 内部二次干扰
            final_test_dataset = RecWithContrastiveLearningDataset(
                args, 
                filtered_test_seq, 
                data_type='test', 
                long_tail_user_set=None 
            )
            
        elif args.test_mode == 'all':
            print(f"🎯 [模式锁死] 当前测试模式为: 【对全体用户做无差别测试 (All-Train, All-Test)】")
            # 全体测试直接使用完整的 test_user_seq
            final_test_dataset = RecWithContrastiveLearningDataset(
                args, 
                test_user_seq, 
                data_type='test', 
                long_tail_user_set=None
            )
            
        else:
            raise ValueError(f"未知测试模式 args.test_mode: {args.test_mode}，必须为 'tail' 或 'all'")

        # 重新绑定底层的 test_dataloader
        trainer.test_dataloader = DataLoader(
            final_test_dataset, 
            sampler=SequentialSampler(final_test_dataset), 
            batch_size=args.batch_size
        )
        print(f"🚀 [大考封盘] 模式 {args.test_mode.upper()} 实际进入 DataLoader 迭代的测试序列数: {len(final_test_dataset)}")
        
        # 触发最终大考离线评测
        final_scores, final_result_info = trainer.test(999, full_sort=True)
        
        # 实时落盘封盘
        print(f"\n🏆 [MSCL-Tail 最终实验总指标 - 模式: {args.test_mode.upper()}]:")
        print(final_result_info)
        
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n================ 🏆 [MSCL-Tail 最终实验总指标 - MODE: {args.test_mode.upper()}] ================\n")
            f.write(f"Model-Data: {args.model_name}-{args.data_name}-{args.model_idx}-Final_Best\n")
            f.write(final_result_info + '\n')
        print(f"✅ 最佳学术指标已成功死死锁进日志: {args.log_file}")

if __name__ == '__main__':
    main()