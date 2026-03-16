import json
import random

import torch
from torch.utils.data import Dataset


class BinaryQADataset(Dataset):
    def __init__(self, dataset_path, tokenizer, max_length=512, is_train=True):
        """
        Args:
            dataset_path: The path to the JSON file containing the training data
            tokenizer: Transformers tokenizer
            max_length: Maximum sequence length
            is_train: Is this training mode (where positive and negative samples are generated during training, and only encoding is performed during testing)?
        """
        datas = []
        for path in dataset_path:
            with open(path, 'r', encoding='utf-8') as f:
                datas.extend(json.load(f))
        self.raw_data = datas
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        # Pre-process the data: create (question, passage, label) samples for training
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

            # Add a positive sample
            processed.append({
                'question': question,
                'context': correct_context,
                'label': 1
            })

            # Add negative samples
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

        # Remove the batch dimension (as each sample is processed individually)
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


