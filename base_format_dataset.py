import json
import random

import torch
from torch.utils.data import Dataset


class BinaryQADataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length=512, is_train=True):
        """
        Args:
            dataset_path: 包含训练数据的JSON文件路径
            tokenizer: Transformers tokenizer
            max_length: 最大序列长度
            is_train: 是否为训练模式（训练时生成正负样本，测试时只编码）
        """
        datas = []
        for path in dataset_path:
            with open(path, 'r', encoding='utf-8') as f:
                datas.extend(json.load(f))
        self.raw_data = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        # 预处理数据：为训练创建(question, passage, label)样本
        self.processed_data = self._preprocess_data()

        self.tokenizer.padding_side = 'right'

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.sep_token is not None:
            self.tokenizer.sep_token = self.tokenizer.eos_token

    def _preprocess_data(self, incor_num=5):
        processed = []

        for i in range(len(self.raw_data)):
            item = self.raw_data[i]

            question = item['question']
            correct_context = item["context"]['correct_context']
            incorrect_context = item["context"]['incorrect_context']

            # 添加正样本
            processed.append({
                'question': question,
                'context': correct_context,
                'label': 1
            })

            # 添加负样本
            processed.append({
                'question': question,
                'context': incorrect_context,
                'label': 0
            })

            if i > len(self.raw_data) / 2:
                range_incor_context_ = random.sample(self.raw_data[:i], incor_num)
                for context in range_incor_context_:
                    processed.append({
                        'question': question,
                        'context': context["context"]['correct_context'],
                        'label': 0
                    })
                    processed.append({
                        'question': question,
                        'context': context["context"]['incorrect_context'],
                        'label': 0
                    })
            else:
                range_incor_context_ = random.sample(self.raw_data[i + 1:], incor_num)
                for context in range_incor_context_:
                    processed.append({
                        'question': question,
                        'context': context["context"]['correct_context'],
                        'label': 0
                    })
                    processed.append({
                        'question': question,
                        'context': context["context"]['incorrect_context'],
                        'label': 0
                    })
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        messages = [
            {"role": "user", "content": f"Can the reference text answer this question?\nReference text:\n{item['context']}\nQuestion:{item['question']}"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("="*30)
        print(text)
        print("-"*30)


        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 移除batch维度（因为每个样本单独处理）
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        if 'token_type_ids' in encoding:
            token_type_ids = encoding['token_type_ids'].squeeze(0)
        else:
            token_type_ids = None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            # 'token_type_ids': token_type_ids,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


""""
可用参考文档数：3
可用参考文档数：26
可用参考文档数：10
可用参考文档数：34
可用参考文档数：21
可用参考文档数：18
可用参考文档数：15
可用参考文档数：18
可用参考文档数：26
可用参考文档数：14
可用参考文档数：11
可用参考文档数：17
可用参考文档数：25
可用参考文档数：7
可用参考文档数：22
可用参考文档数：7
可用参考文档数：7
可用参考文档数：9
可用参考文档数：18
可用参考文档数：12
可用参考文档数：21
可用参考文档数：21
可用参考文档数：28
可用参考文档数：20
可用参考文档数：14
可用参考文档数：21
可用参考文档数：7
可用参考文档数：14
可用参考文档数：12
"""