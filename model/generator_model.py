import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class Generator:
    def __init__(self, base_model_path: str, tokenizer_path: str, gpu_idx:int, lora_path: str) -> None:
        self.device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        self.base_model_path = base_model_path
        if not lora_path:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=base_model_path,
                torch_dtype=torch.float16,
            ).to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=base_model_path,
                torch_dtype=torch.float16,
            )
            self.model = PeftModel.from_pretrained(self.model, lora_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )

    def generate(self, question, reference_list, split_tokens:str):
        prompt = [
            {"role": "user" if "deepseek" in self.base_model_path.lower() else "system",
             "content": "Your task is to answer the specified question based on the given reference text."
                        " If the reference text cannot answer the question, please ignore the reference text and try to answer independently. "
                        "If you cannot answer due to a lack of corresponding knowledge or other reasons, output 'I don't know.'."
                        "Simply provide the complete question response or output “I don't know.” No additional explanations, science popularization, or other content is required. "
                        "Ensure your response remains concise."
                        "Your response should be no more than 50 words."
                        "Don't just provide the answer; you should integrate the answer with the question in your response."
                        f"Place your reply after {split_tokens[1]} ."
                        f"""Example 1:
                                             Question: Who is eating bread?
                                             Reference Document: Zhang San had been hungry for days. While visiting his neighbor Li Si, he saw Li Si eating bread and wanted to share it, but Li Si refused.
                                             Your Response:{split_tokens[1]} Li Si is eating bread.

                                         Example 2:
                                             Question: Who is the author of the song “Chichi”?
                                             Reference Document: Xue Zhiqian is a top singer in mainland China. He released the song “Chichi” last year, which received an exceptionally positive response. Everyone loves this song.
                                             Your Response:{split_tokens[1]} The author of the song “Chichi” is Xue Zhiqian.

                                         Example 3:
                                             Question: Who invented papermaking?
                                             Reference Document: No one knows where Wang Wei is, nor what he likes to eat. We only know that he invented the airplane.
                                             Your Response:{split_tokens[1]} I don't know."""},
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
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=2000,
                temperature=0.6,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id
                # return_prompt=False
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split(split_tokens[0])[-1].strip().split(split_tokens[1])[-1].strip()