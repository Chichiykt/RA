import argparse
import heapq
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model.clean_model import UtilModel
from model.nli_model import NLIModel
from model.text_matching_model import TextMatchingModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,4"
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


class Generator:
    def __init__(self, generator_path: str):
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        # 加载模型和分词器
        self.model = AutoModelForCausalLM.from_pretrained(generator_path)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_path, device_map={"":"cuda:2"})
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
                                             Your Response: Li Si is eating bread.

                                         Example 2:
                                             Question: Who is the author of the song “Chichi”?
                                             Reference Document: Xue Zhiqian is a top singer in mainland China. He released the song “Chichi” last year, which received an exceptionally positive response. Everyone loves this song.
                                             Your Response: The author of the song “Chichi” is Xue Zhiqian.

                                         Example 3:
                                             Question: Who invented papermaking?
                                             Reference Document: No one knows where Wang Wei is, nor what he likes to eat. We only know that he invented the airplane.
                                             Your Response: I don't know."""},
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
                max_new_tokens=4096,
                # num_return_sequence=1,
                # temperature=0.7,
                # do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # return_prompt=False
            )
            # new_tokens = outputs.sequences[0, len(inputs["input_ids"]):]
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("</think>")[-1].strip()

class NliModel:
    def __init__(self, model_path):
        self.nli = NLIModel(model_path)
    def predict_generator_answer(self, premise, question, answer):
        return self.nli.predict(premise, f"The answer to '{question}' is '{answer}'.")

    def check_IDK(self, generate_answer):
        unknown_template = """An effective response must provide substantive information relevant to the question.
                 If the response indicates that the model is unable to provide an answer due to knowledge limitations, capability constraints, or strategic reasons, it is considered invalid. 
                Such responses typically include (but are not limited to) the following expressions: 
                “I don't know,” “I don't have the relevant information,” “I cannot answer,” “As an AI model...,” “Based on my knowledge base...,” 
                or evading the question with irrelevant content."""
        return self.nli.predict(unknown_template, generate_answer)
    # def predict(self, premise, hypothesis):
    #     return self.nli.predict(premise, hypothesis)


if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))
    params = argparse.ArgumentParser()
    params.add_argument('--eval_type', type=str,
                        help='eval base model :base_model',)
    params.add_argument('--test_data', type=str,
                        default='D:\\AI_datasets\\MNIST_Dataset\\MNIST_Dataset\\test_images',
                        help='the root path of the test data')
    params.add_argument('--batch_size', type=int, default=64)
    params.add_argument('--epochs', type=int, default=66)
    params.add_argument('--learning_rate', type=float, default=0.01)
    params.add_argument('--dir_to_label', type=bool, default=True, help='the name of dir is the label that in this dir')
    params.add_argument('--name_to_label', type=bool, default=True,
                        help='the name of every image contain the target label')
    params.add_argument('--split_char', type=str, default='_',
                        help='use with --name_to_label, can get accuracy label by this split char')
    params.add_argument('--label_dir', type=str, default='')

    args = params.parse_args()
    generator_path = os.path.join(current_dir, "llama3_8b_instruct")

    origin_data_path = os.path.join(current_dir, "download_nq-open_trivia-qa/open_domain_data/NQ/train.json")
    save_path = os.path.join(current_dir, "dataset/nq_open_domain.json")
    nli_model_path = os.path.join(current_dir, "tasksource_deberta_small_long_nli")

    matching_base_model_path = os.path.join(current_dir, "deepSeek_R1_Distill_Qwen_1.5B")
    lora_weight_path = os.path.join(current_dir, "text_matching_model/last_model_lora")
    matching_dict_path = os.path.join(current_dir, "text_matching_model/last_model_dict/model_dict.bin")
    matching_tokenizer_path = os.path.join(current_dir, "text_matching_model/last_model_tokenizer")
    clean_model_path = os.path.join(current_dir, "llama3_8b_instruct")

    context_clean_model = UtilModel(model_path=clean_model_path, tokenizer=None, gpu_idx=0,
                                         tokenizer_path=clean_model_path)

    matching_model = TextMatchingModel(base_model_path=matching_base_model_path, model_dict_path=matching_dict_path,
                                       matching_tokenizer_path=matching_tokenizer_path,
                                       context_clean_model=context_clean_model, lora_weight_path=lora_weight_path)

    top_n = TopNFloats(50)
    generator = Generator(generator_path)

    nli_model = NliModel(nli_model_path)

    with open(origin_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    data_case = {
        'question': "",
        'answer':"",
        'contexts':""
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        question_count = len(raw_data)
        right_count = 0

        datas = []
        for data in raw_data:
            question = data['question']
            answer = data['answers']
            data_case['question'] = question
            data_case['answer'] = answer
            IDK_count = 0
            R_count = 0
            for references in data['ctxs']:
                reference = references["text"]
                a, b = matching_model.predict(question, reference)
                # print("问题:", question, "prob", b, "参考文档：", reference, "答案：", answer, sep='\n', end='\n')
                top_n.add(reference, b)
            data_case['contexts'] = top_n.get_top_n()

            generator_answer = generator.generate(question=question, reference_list=data_case['contexts'])
            print("模型回复:", generator_answer)
            result, probs = check_idk = nli_model.check_IDK(generator_answer)
            if result == "entailment":
                print(f"(我不知道)问题：{question}", f"答案：{answer}", f"模型回复：{generator_answer}" ,"==="*20,sep='\n', end='\n')
                continue

            answer_str = data_case['answer'][0]
            for ans in data_case['answer'][1:]:
                answer_str += f",{ans}"


            result, probs = nli_model.predict_generator_answer(premise=generator_answer, question=question, answer=answer_str)
            if result == "entailment":
                for ans in data_case['answer']:
                    if ans not in generator_answer:
                        continue
                right_count += 1
                print(f"(正确回答)问题：{question}" , f"答案：{answer}" , f"模型回复：{generator_answer}", sep='\n', end='\n')
            else:
                print(f"问题：{question}", f"答案：{answer}", f"模型回复：{generator_answer}", "="*20, sep='\n', end='\n')

            datas.append(data_case)
            json.dump(datas, f, ensure_ascii=True, indent=4)


