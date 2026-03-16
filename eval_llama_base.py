import heapq
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.clean_model import UtilModel
from model.nli_model import NLIModel

from model.text_matching_model import TextMatchingModel

class Generator:
    def __init__(self, generator_path: str, gpu_idx: int) -> None:
        self.device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(generator_path)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_path, device_map={"":f"cuda:{gpu_idx}"})
        self.model.to(device=self.device)
        print("生成器模型加载成功！")

    def generate(self, question, reference_list):
        prompt = [
            {"role": "user",
             "content": "Your task is to answer the specified question based on the given reference text."
                        " If the reference text cannot answer the question, please ignore the reference text and try to answer independently. "
                        "If you cannot answer due to a lack of corresponding knowledge or other reasons, output 'I don't know.'."
                        "Simply provide the complete question response or output “I don't know.” No additional explanations, science popularization, or other content is required. "
                        "Ensure your response remains concise."
                        "Your response should be no more than 50 words."
                        "Don't just provide the answer; you should integrate the answer with the question in your response."
                        """Example 1:
                                             Question: Who is eating bread?
                                             Reference Document: Zhang San had been hungry for days. While visiting his neighbor Li Si, he saw Li Si eating bread and wanted to share it, but Li Si refused.
                                             Your Response:Li Si is eating bread.

                                         Example 2:
                                             Question: Who is the author of the song “Chichi”?
                                             Reference Document: Xue Zhiqian is a top singer in mainland China. He released the song “Chichi” last year, which received an exceptionally positive response. Everyone loves this song.
                                             Your Response:The author of the song “Chichi” is Xue Zhiqian.

                                         Example 3:
                                             Question: Who invented papermaking?
                                             Reference Document: No one knows where Wang Wei is, nor what he likes to eat. We only know that he invented the airplane.
                                             Your Response:I don't know."""},
            {"role": "user", "content": f"Question: {question}?"},
            {"role": "user",
             "content": "\n".join([f"Reference {i + 1}: {ref}" for i, ref in enumerate(reference_list)])}
        ]

        context = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(context, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                inputs["attention_mask"],
                max_new_tokens=512,
                # num_return_sequence=1,
                # temperature=0.7,
                # do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # return_prompt=False
            )
            # new_tokens = outputs.sequences[0, len(inputs["input_ids"]):]
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("<Answer>:")[-1].strip()
class NliModel:
    def __init__(self, model_path):
        self.nli = NLIModel(model_path)

    def predict_generator_answer(self, model_answer, standard_answer):
        return self.nli.predict(model_answer, standard_answer)

    def check_IDK(self, generate_answer):
        unknown_template = """An effective response must provide substantive information relevant to the question.
                 If the response indicates that the model is unable to provide an answer due to knowledge limitations, capability constraints, or strategic reasons, it is considered invalid. 
                Such responses typically include (but are not limited to) the following expressions: 
                “I don't know,” “I don't have the relevant information,” “I cannot answer,” “As an AI model...,” “Based on my knowledge base...,” 
                or evading the question with irrelevant content."""
        return self.nli.predict(unknown_template, generate_answer)
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

def main():
    #1.数据模型路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    origin_data_path = os.path.join(current_dir, "download_nq-open_trivia-qa/open_domain_data/NQ/train.json")

    generator_path = os.path.join(current_dir, "llama3_8b_instruct")

    matching_base_model_path = os.path.join(current_dir, "deepSeek_R1_Distill_Qwen_1.5B")
    matching_lora_weight_path = os.path.join(current_dir, "text_matching_model/last_model_lora")
    matching_dict_path = os.path.join(current_dir, "text_matching_model/last_model_dict/model_dict.bin")
    matching_tokenizer_path = os.path.join(current_dir, "text_matching_model/last_model_tokenizer")

    data_save_path = os.path.join(current_dir, "dataset/nq_open_domain.json")
    #文本优化模型
    clean_model_path = os.path.join(current_dir, "llama3.2_3b_instruct")

    nli_model_path = os.path.join(current_dir, "tasksource_deberta_small_long_nli")

    nli_model = NliModel(nli_model_path)
    #2.加载数据模型
    with open(origin_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    util_model = UtilModel(model_path=clean_model_path, tokenizer=None, gpu_idx=3,
                                         tokenizer_path=clean_model_path)
    generator = Generator(generator_path, gpu_idx=2)
    matching_model = TextMatchingModel(base_model_path=matching_base_model_path, model_dict_path=matching_dict_path,
                                       matching_tokenizer_path=matching_tokenizer_path,
                                       context_clean_model=util_model, lora_weight_path=matching_lora_weight_path)
    datas = []
    data_case = {
        'question': "",
        'answer': "",
        'contexts': ""
    }
    top_n = TopNFloats(50)
    ed = 0
    with open(data_save_path, 'w', encoding='utf-8') as f:
        for i, data in enumerate(raw_data):
            print(f"当前问题：{i+1}/{len(raw_data)}")
            question = data['question']
            answer = data['answers']
            data_case['question'] = question
            data_case['answer'] = answer
            IDK_count = 0
            R_count = 0
            print("筛选参考文档...")
            for i, references in enumerate(data['ctxs']):
                reference = references["text"]
                a, b = matching_model.predict(question, reference)
                # print("问题:", question, "prob", b, "参考文档：", reference, "答案：", answer, sep='\n', end='\n')
                top_n.add(reference, b)
                print(f"{i+1}/{len(data['ctxs'])}")
            print("top N文档筛选完成")
            data_case['contexts'] = top_n.get_top_n()

            model_answer = generator.generate(question=question, reference_list=data_case['contexts'])
            standard_answer = util_model.question_answer_combine(question=question, answer=answer)

            if "entailment" == nli_model.check_IDK(model_answer):
                print("(我不知道)问题:", question, "模型回复：", model_answer,sep='\n', end='\n')
                IDK_count += 1
                continue
            if "entailment" == nli_model.predict_generator_answer(model_answer=model_answer, standard_answer=standard_answer):
                print("(回答正确)问题:", question, "答案：", answer,"模型回复：", model_answer, "标准回复：", standard_answer, sep="\n", end="\n")
                R_count += 1
                continue
            print("问题:", question, "模型回复：", model_answer, "标准回复：", standard_answer, sep="\n", end="\n")
            datas.append(data_case.copy())
            ed += 1
            print("--------------------当前: ", f"IDK: {IDK_count}/{ed}", f"R_count: {R_count}/{ed}")
        json.dump(datas, f, ensure_ascii=True, indent=4)
if __name__ == '__main__':
    main()








