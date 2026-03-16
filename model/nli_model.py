import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIModel:
    """自然语言推理模型，判断答案是否正确"""
    def __init__(self, model_path: str, gpu_idx: int):
        self.device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.max_length = self.model.config.max_position_embeddings
        self.model.eval()

    def predict(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            print("premise: ", premise, "\nhypothesis: ", hypothesis , "\npredictions: ", predictions, "\npredicted_class: ", predicted_class)
            # 根据模型输出映射到标签
            label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
            return label_map[predicted_class], predictions[0][predicted_class].cpu().numpy().item()

