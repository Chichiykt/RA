import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from clean_model import CleanModel
import os

class TextMatchingModel:
    """Text matching model to determine whether a question can be answered"""

    def __init__(self, base_model_path, model_dict_path, matching_tokenizer_path,context_clean_model: CleanModel, lora_weight_path, device_index):
        self.context_cleaner = context_clean_model
        self.device_index = device_index

        self.device = torch.device(f"cuda:{self.device_index}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(matching_tokenizer_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_type_id = self.tokenizer.eos_token_id


        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=2,
            problem_type="single_label_classification",
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(model, lora_weight_path).to(self.device)
        self.model.load_state_dict(torch.load(model_dict_path, map_location=self.device))

        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        print("The model has finished loading")

    def predict(self, question: str, reference_text: str):
        cleaned_ref_text = self.context_cleaner.clean_context(reference_text)
        messages = [
            {"role": "user",
             "content": f"Can the reference text answer this question?\nReference text:\n{cleaned_ref_text}\nQuestion:{question}"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        with torch.no_grad():
            inputs.to(self.device)
            outputs = self.model(**inputs)
        return torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()[1] > 0.5, torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()[1]
