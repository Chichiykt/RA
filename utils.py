import heapq
import json
import random

"""
Retain the top N documents with the highest scores
"""
class TopNFloats:
    def __init__(self, n):
        self.n = n
        self.heap = []  

    def add(self, text, prob):
        if len(self.heap) < self.n:
            heapq.heappush(self.heap, (text, prob))
        else:
            if prob > self.heap[0][1]:  
                heapq.heapreplace(self.heap, (text, prob))

    def get_top_n(self):
        return [i[0] for i in sorted(self.heap, reverse=True)]

"""
There are datasets from multiple reference documents
"""
class multi_context_dataset:
    def __init__(self, dataset_path:str, context_count: int, correct_probability:float):
        """
        :param dataset_path:
        :param context_count: Number of reference texts per question
        :param correct_probability: The proportion of questions with the correct reference documentation
        """
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        self.context_count = context_count
        self.correct_probability = correct_probability
        self.data_all = self._preprocess_data()

    def random_bool(self, true_probability):

        if not 0 <= true_probability <= 1:
            raise ValueError("The probability must be between 0 and 1")

        return random.random() < true_probability
    def _preprocess_data(self, count:int=960) -> list:
        data_all = []
        contexts = []
        for row in self.raw_data:
            contexts.append(row["context"]["correct_context"])
            contexts.append(row["context"]["incorrect_context"])

        for i in range(len(self.raw_data)):
            item = self.raw_data[i]
            data_single = {"question": item["question"] + '?',
                           "contexts": [],
                           "answer": item["answer"],
                           "correct_context": item["context"]['correct_context']}

            if self.random_bool(self.correct_probability):
                data_single["contexts"].append(item["context"]['correct_context'])
                data_single["contexts"].append(item["context"]['incorrect_context'])

            if i * 2 > len(contexts) / 2:
                data_single["contexts"].extend(
                    random.sample(contexts[: i * 2], self.context_count - len(data_single["contexts"])))
            else:
                data_single["contexts"].extend(
                    random.sample(contexts[i * 2 + 2:], self.context_count - len(data_single["contexts"])))

            random.shuffle(data_single["contexts"])

            data_all.append(data_single.copy())
        return data_all[:count]
