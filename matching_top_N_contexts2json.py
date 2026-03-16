import json
import os
import heapq

from model.text_matching_model import TextMatchingModel

current_dir = os.path.dirname(os.path.abspath(__file__))
origin_data_path = os.path.join(current_dir, "download_nq-open_trivia-qa/open_domain_data/NQ/train.json")
save_path = os.path.join(current_dir, "dataset/nq_open_domain.json")
matching_model_path = os.path.join(current_dir, "deepSeek_R1_Distill_Qwen_1.5B")
lora_weight_path = os.path.join(current_dir, "text_matching_model/best_model")

matching_model = TextMatchingModel(matching_model_path, lora_weights_path=lora_weight_path)

class TopNFloats:
    def __init__(self, n):
        self.n = n
        self.heap = []  # 最小堆，保存最大的N个数

    def add(self, text, prob):
        if len(self.heap) < self.n:
            heapq.heappush(self.heap, (text, prob))
        else:
            if prob > self.heap[0][1]:  # 如果比堆顶大
                heapq.heapreplace(self.heap, (text, prob))

    def get_top_n(self):
        # 返回排序后的前N个（从大到小）
        return [i[0] for i in sorted(self.heap, reverse=True)]

top_n = TopNFloats(20)

with open(origin_data_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

data_case = {
    'question': "",
    'answer':"",
    'contexts':""
}

with open(save_path, 'w', encoding='utf-8') as f:

    datas = []
    for data in raw_data:
        question = data['question']
        answer = data['answers']
        data_case['question'] = question
        data_case['answer'] = answer
        count = 0
        for references in data['ctxs']:
            reference = references["text"]
            a, b = matching_model.predict(question, reference)
            # print("问题:", question, "prob", b, "参考文档：", reference, "答案：", answer, sep='\n', end='\n')
            top_n.add(reference, b)
        data_case['contexts'] = top_n.get_top_n()
        datas.append(data_case)
        print(1)
    json.dump(datas,f, ensure_ascii=True,indent=4)

    # print("1")

    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=True, indent=4)