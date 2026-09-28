import random
import torch
from torch.utils.data import Dataset
from data_augmentation import Crop, Mask, Reorder, Substitute, Insert, Random
from utils import neg_sample
from utils import nCr
import copy


class RecWithContrastiveLearningDataset(Dataset):
    def __init__(self, args, user_seq, data_type='train',
                 similarity_model_type='offline', long_tail_user_set=None):

        self.args = args
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.similarity_model = args.offline_similarity_model

        # 筛选 user_seq：只保留长尾用户（仅在 valid/test 阶段） and self.args.tail_test
        '''if data_type in ['valid', 'test'] and long_tail_user_set is not None:
            self.user_seq = [
                seq for uid, seq in enumerate(user_seq)
                if uid in long_tail_user_set and len(seq) >= 3
            ]
        else:'''
        self.user_seq = [seq for seq in user_seq if len(seq) >= 3]

        print(f"✅ {data_type.upper()} 集共包含序列数: {len(self.user_seq)}")

        print("Similarity Model Type:", similarity_model_type)
        self.augmentations = {
            'crop': Crop(tao=args.tao),
            'mask': Mask(gamma=args.gamma),
            'reorder': Reorder(beta=args.beta),
            'substitute': Substitute(self.similarity_model, substitute_rate=args.substitute_rate),
            'insert': Insert(self.similarity_model, insert_rate=args.insert_rate),
            'random': Random(
                tao=args.tao,
                gamma=args.gamma,
                beta=args.beta,
                item_similarity_model=self.similarity_model,
                insert_rate=args.insert_rate,
                max_insert_num_per_pos=args.max_insert_num_per_pos,
                substitute_rate=args.substitute_rate,
                augment_combination=args.random_augment_combination
            )
        
        }

        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]
        self.n_views = self.args.n_views


    def _one_pair_data_augmentation(self, input_ids):
        '''
        provides two positive samples given one sequence
        '''

        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids, model=self.args.model)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len

            cur_tensors = (
                torch.tensor(augmented_input_ids, dtype=torch.long)
            )
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)
        copied_input_ids = [0] * pad_len + copied_input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(copied_input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]

        # =========================================================================
        # 🍏 👑 【完美多维对齐防御罩】：遇到被抹平的非长尾用户，做尺寸对齐假动作
        # =========================================================================
        if len(items) == 0:
            # 1. 动态获取你模型配置的固定序列 Padding 长度（从报错看是 50，这里做鲁棒性兼容）
            max_len = getattr(self.args, 'max_seq_len', 50) 
            
            # 2. 构建一个长度完全对齐的假输入 Tensor（全填0，充当占位符）
            fake_input_ids = torch.zeros(max_len, dtype=torch.long)
            
            return (
                torch.tensor(user_id, dtype=torch.long),
                fake_input_ids,                         # 👑 动作对齐：返回长度为 50 的全零占位符
                torch.tensor(0, dtype=torch.long),      # 标量 0 
                torch.tensor(0, dtype=torch.long),      # 标量 0
                torch.tensor(-1, dtype=torch.long)     # 👑 灵魂哨兵：答案依然是 -1，代表缺考
            )
        # =========================================================================
            
        # 允许传入 "pure_train"
        assert self.data_type in {"train", "valid", "test", "pure_train"}

        # 👑 新增：接收在 main.py 里已经提纯完、无考题、无交叉的纯粹独立样本池
        if self.data_type == "pure_train":
            # 此时的 items 已经是纯训练前缀或纯独立伪序列，首尾错开一位形成推荐系统的预测 Target 即可
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [0] # 训练不需要 answer
            
        elif self.data_type == "train":
            # Stage 1 原生预训练的扣除逻辑保持不变
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]
             
            
            
        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        # test
        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        if self.data_type == "train":
            # 保持你原有的 padding 和截断逻辑
            pad_len = max(0, self.max_len - len(input_ids))
            input_ids = [0] * pad_len + input_ids
            target_pos = [0] * pad_len + target_pos

            input_ids = input_ids[-self.max_len:]
            target_pos = target_pos[-self.max_len:]

            # 统一构造成长尾模型训练所需的标准 3 元组张量
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
            )
            return cur_rec_tensors
        elif self.data_type == 'valid':
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, \
                                                         target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, \
                                                         target_pos, answer)
            return cur_rec_tensors
        for uid, seq in enumerate(test_seq[:5]):
            print(f"用户 {uid} 的 test 序列: {seq}")
        if len(seq) >= 1:
            print("   👉 预测目标是:", seq[-1])

    def __len__(self):
        '''
        consider n_view of a single sequence as one sample
        '''
        
        return len(self.user_seq)


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, \
                                          target_pos, answer)

    def __len__(self):
        return len(self.user_seq)
