import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding, AutoTokenizer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
from base_format_dataset import BinaryQADataset


def compute_metrics(eval_pred):
    """Calculate evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    # Calculate the AUC
    probs = torch.softmax(torch.tensor(eval_pred.predictions), dim=1)
    auc = roc_auc_score(labels, probs[:, 1].numpy())

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }


# def collate_fn(batch):
#     input_ids = torch.stack([item['input_ids'] for item in batch])
#     attention_mask = torch.stack([item['attention_mask'] for item in batch])
#     labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
#
#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'labels': labels
#     }


class TextMatchingTrainer:
    def __init__(self, model_path, dataset_path, output_dir):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            problem_type="single_label_classification"
        )

        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  
            inference_mode=False,
            r=16,  
            lora_alpha=32,  
            lora_dropout=0.1,  
            target_modules=["q_proj", "v_proj"], 
            modules_to_save=["score"]
        )

        self.model = get_peft_model(self.base_model, self.lora_config)

        self.model.print_trainable_parameters()

    def prepare_datasets(self, test_size=0.2):
        full_dataset = BinaryQADataset(
            self.dataset_path, self.tokenizer
        )
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))
        split_idx = int(dataset_size * (1 - test_size))

        train_indices = indices[:split_idx]
        eval_indices = indices[split_idx:]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)

        print(f"Total sample size: {dataset_size}")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def train(self, epochs, batch_size, learning_rate=2e-4): 
        """Using a LoRA fine-tuned model"""
        # Prepare the data
        train_dataset, eval_dataset = self.prepare_datasets()

        # Training parameters – Optimised for LoRA
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_steps=100,  # LoRA training typically requires less pre-training
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=50,  
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            report_to=None,
            gradient_accumulation_steps=2,  
            optim="adamw_torch",  
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )

        # Start training
        print("Get started with fine-tuning text-matching models using LoRA...")
        trainer.train()

        # Save the best model – save only the LoRA weights
        lase_model_dict_path = os.path.join(self.output_dir, "last_model_dict/model_dict.bin")
        lase_model_tokenizer_path = os.path.join(self.output_dir, "last_model_tokenizer")
        lase_model_lora_path = os.path.join(self.output_dir, "last_model_lora")
        trainer.model.save_pretrained(lase_model_lora_path)

        torch.save(trainer.model.state_dict(),lase_model_dict_path)
        self.tokenizer.save_pretrained(lase_model_tokenizer_path)

        print(f"LoRA fine-tuning is complete; the model weights have been saved to: {lase_model_dict_path}")
        return trainer

    def load_lora_model(self, lora_path):
        """Load pre-trained LoRA weights"""
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.model.eval()
        print("LoRA weights have been loaded")

    def predict(self, question, contexts):
        """Predict which text is most likely to answer the question"""
        self.model.eval()
        self.model.to(self.device)

        scores = []
        for context in contexts:
            encoding = self.tokenizer(
                question,
                context,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )

            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**encoding)
                probs = torch.softmax(outputs.logits, dim=1)
                match_score = probs[0][1].item()  # Probability of the positive class

            scores.append(match_score)

        # Return the sorted results
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []

        for idx in sorted_indices:
            results.append({
                'context': contexts[idx],
                'score': scores[idx],
                'rank': len(results) + 1
            })

        return results

    def merge_and_save_full_model(self, output_path):
        """Merge the LoRA weights into the base model and save the complete model"""
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print(f"The complete model has been saved to: {output_path}")
        return merged_model

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "deepSeek_R1_Distill_Qwen_1.5B")
    # Configuration parameters
    # model_path = "./DeepSeek_R1_Distill_Qwen_1.5B"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = [os.path.join(current_dir, "dataset/nq_open_train.json"), os.path.join(current_dir, "dataset/nq_open__val.json")]
    # dataset_path = "./datasets/nq_open_train.json"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "text_matching_model")
    # output_dir = "./text_matching_model"

    trainer = TextMatchingTrainer(model_path, dataset_path, output_dir)
    trainer.train(epochs=3, batch_size=8, learning_rate=2e-4)



if __name__ == "__main__":
    main()

"""

{'eval_loss': 0.05197696015238762, 'eval_accuracy': 0.9891666666666666, 'eval_f1': 0.9356435643564357, 'eval_precision': 0.9264705882352942, 'eval_recall': 0.945, 'eval_auc': 0.9977590909090908, 'eval_runtime': 79.6044, 'eval_samples_per_second': 30.149, 'eval_steps_per_second': 1.884, 'epoch': 3.0}                                                                                                      
{'train_runtime': 2606.4301, 'train_samples_per_second': 11.05, 'train_steps_per_second': 0.691, 'train_loss': 0.13474373790952895, 'epoch': 3.0}                                                        
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [43:26<00:00,  1.45s/it]
LoRA Fine-tuning complete! The model has been saved to: /tmp/pycharm_project_679/text_matching_model/best_model
LoRA The size of the weight file is significantly smaller than that of the full model
"""
