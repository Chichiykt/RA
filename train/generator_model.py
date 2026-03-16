import torch
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

class GeneratorModel:

    def __init__(self, model_path: str, gpu_idx: int, tokenizer_path: str=None) -> None:
        self.device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            target_modules=["q_proj", "v_proj"]
        )

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path,
                                                          torch_dtype=torch.float16,
                                                          peft_config=self.lora_config
                                                          ).to(self.device)

    def __create_prompt(self, question: str, contexts: list):
        return [
            {"role": "user",
             "content": "Your task is to answer the specified question based on the given reference text."
                        " If the reference text cannot answer the question, please ignore the reference text and try to answer independently. "
                        "If you cannot answer due to a lack of corresponding knowledge or other reasons, output 'I don't know.'."
                        "Simply provide the complete question response or output “I don't know.” No additional explanations, science popularization, or other content is required. "
                        "Ensure your response remains concise."
                        "Your response should be no more than 50 words."
                        "Don't just provide the answer; you should integrate the answer with the question in your response."
                        "Place your reply after <Answer>: ."
                        """Example 1:
                                             Question: Who is eating bread?
                                             Reference Document: Zhang San had been hungry for days. While visiting his neighbor Li Si, he saw Li Si eating bread and wanted to share it, but Li Si refused.
                                             Your Response:<Answer>: Li Si is eating bread.

                                         Example 2:
                                             Question: Who is the author of the song “Chichi”?
                                             Reference Document: Xue Zhiqian is a top singer in mainland China. He released the song “Chichi” last year, which received an exceptionally positive response. Everyone loves this song.
                                             Your Response:<Answer>: The author of the song “Chichi” is Xue Zhiqian.

                                         Example 3:
                                             Question: Who invented papermaking?
                                             Reference Document: No one knows where Wang Wei is, nor what he likes to eat. We only know that he invented the airplane.
                                             Your Response:<Answer>: I don't know."""},
            {"role": "user", "content": f"Question: {question}?"},
            {"role": "user",
             "content": "\n".join([f"Reference {i + 1}: {ref}" for i, ref in enumerate(contexts)])}
        ]
"""
    Your task is to answer the specified question based on the given reference text.
    If the reference text cannot answer the question, please ignore the reference text and try to answer independently. 
    If you cannot answer due to a lack of corresponding knowledge or other reasons, output 'I don't know.'.
    Simply provide the complete question response or output “I don't know.” No additional explanations, science popularization, or other content is required. 
    Ensure your response remains concise.
    Your response should be no more than 50 words.
    Don't just provide the answer; you should integrate the answer with the question in your response.
    Place your reply after <Answer>: .
    Example 1:
        Question: Who is eating bread?
        Reference Document: Zhang San had been hungry for days. While visiting his neighbor Li Si, he saw Li Si eating bread and wanted to share it, but Li Si refused.
        Your Response:<Answer>: Li Si is eating bread.
    Example 2:
        Question: Who is the author of the song “Chichi”?
        Reference Document: Xue Zhiqian is a top singer in mainland China. He released the song “Chichi” last year, which received an exceptionally positive response. Everyone loves this song.
        Your Response:<Answer>: The author of the song “Chichi” is Xue Zhiqian.
    Example 3:
        Question: Who invented papermaking?
        Reference Document: No one knows where Wang Wei is, nor what he likes to eat. We only know that he invented the airplane.
        Your Response:<Answer>: I don't know.
    
    """


    def get_input_ids(self, question: str,contexts: list):
        inputs_prompt = self.__create_prompt(question=question, contexts=contexts)

        inputs_prompt = self.tokenizer.apply_chat_template(
            inputs_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(inputs_prompt, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs["input_ids"][0]