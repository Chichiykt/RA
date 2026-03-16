import json
import os
import random
# from log_function import logger
import torch
from torch.utils.data import Dataset

class MultiContextDataset(Dataset):
    def __init__(self, dataset_path: str, count: int,  tokenizer , correct_probability=0.9, max_length=512):
        """
        Args:
            dataset_path: The path to the JSON file containing the training data
            count: The number of samples for each question
            has_correct: Does it contain the correct reference document? true: Yes
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.count = count
        self.correct_probability = correct_probability
        # Pre-process the data: create (question, passage, label) samples for training
        self.data_all = self._preprocess_data()
        # self.tokenizer = tokenizer
        # self.max_length = max_length

        # self.tokenizer.padding_side = 'right'

        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        # if self.tokenizer.sep_token is not None:
        #     self.tokenizer.sep_token = self.tokenizer.eos_token

    def random_bool(self, true_probability):

        if not 0 <= true_probability <= 1:
            raise ValueError("The probability must be between 0 and 1")

        return random.random() < true_probability

    def _preprocess_data(self):
        data_all = []
        contexts = []
        for row in self.raw_data:
            contexts.append(row["context"]["correct_context"])
            contexts.append(row["context"]["incorrect_context"])

        for i in range(len(self.raw_data)):
            item = self.raw_data[i]
            data_single = {"question": item["question"]+'?', "contexts": [], "correct_context":item["context"]['correct_context']}

            if self.random_bool(self.correct_probability):
                data_single["contexts"].append(item["context"]['correct_context'])

                data_single["contexts"].append(item["context"]['incorrect_context'])

            if i * 2 > len(contexts) / 2:
                data_single["contexts"].extend(random.sample(contexts[ : i * 2], self.count - len(data_single["contexts"])))
            else:
                data_single["contexts"].extend(random.sample(contexts[i * 2 + 2 : ], self.count - len(data_single["contexts"])))

            random.shuffle(data_single["contexts"])

            data_all.append(
            {
                "prompt": "Your task is to answer the specified question based on the given reference text."
                          " If the reference text cannot answer the question, please ignore the reference text and try to answer independently. "
                          "If you cannot answer due to a lack of corresponding knowledge or other reasons, output 'I don't know.'."
                          "Simply provide the complete question response or output “I don't know.” No additional explanations, science popularization, or other content is required. "
                          "Ensure your response remains concise."
                          "Your response should be no more than 50 words."
                          "Don't just provide the answer; you should integrate the answer with the question in your response."
                          """Example 1:
                                Question: Who is eating bread?
                                Reference Document: Zhang San had been hungry for days. While visiting his neighbor Li Si, he saw Li Si eating bread and wanted to share it, but Li Si refused.
                                Your Response: Li Si is eating bread.
                                
                            Example 2:
                                Question: Who is the author of the song “Chichi”?
                                Reference Document: Xue Zhiqian is a top singer in mainland China. He released the song “Chichi” last year, which received an exceptionally positive response. Everyone loves this song.
                                Your Response: The author of the song “Chichi” is Xue Zhiqian.
                                
                            Example 3:
                                Question: Who invented papermaking?
                                Reference Document: No one knows where Wang Wei is, nor what he likes to eat. We only know that he invented the airplane.
                                Your Response: I don't know.""",
                "question": f"{data_single['question']}",
                "reference_texts": [context for context in data_single['contexts']],
                "correct_context": data_single["correct_context"]
            })

        return data_all[:960]

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, idx):
        return self.data_all[idx]
    # def __getitem__(self, idx):
    #     item = self.data_all[idx]

        # messages = [
        #     {"role": "user", "content": ""+
        #                                 "\nReference text:\n"+
        #                                 "".join([f"{context}\n" for context in item['contexts']])+
        #                                 f"\nQuestion:\n{item['question']}"}
        # ]

        # logger.info("=" * 30)
        # logger.info(messages)
        # logger.info("-" * 30)

        # text = self.tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        #
        # # Tokenize
        # encoding = self.tokenizer(
        #     text,
        #     truncation=True,
        #     padding='max_length',
        #     max_length=self.max_length,
        #     return_tensors='pt'
        # )
        #
        # # Remove the batch dimension (as each sample is processed individually)
        # input_ids = encoding['input_ids'].squeeze(0)
        # attention_mask = encoding['attention_mask'].squeeze(0)
        #
        # if 'token_type_ids' in encoding:
        #     token_type_ids = encoding['token_type_ids'].squeeze(0)
        # else:
        #     token_type_ids = None
        #
        # return {
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     # 'token_type_ids': token_type_ids,
        #     'labels': torch.tensor(item['label'], dtype=torch.long)
        # }

# if __name__ == '__main__':
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     data_path = os.path.join(current_dir, "dataset/nq_open__val.json")
#     m = MultiContextDataset(data_path, 5, None)
#     logger.info(m.data_all[:10])
