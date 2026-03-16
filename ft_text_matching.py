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
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )

    # 计算AUC
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
#     """数据整理函数"""
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

        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基础模型
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            problem_type="single_label_classification"
        )

        # 配置LoRA
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # 序列分类任务
            inference_mode=False,
            r=16,  # LoRA秩
            lora_alpha=32,  # LoRA alpha参数
            lora_dropout=0.1,  # Dropout率
            target_modules=["q_proj", "v_proj"],  # 目标模块
            modules_to_save=["score"]
        )

        # 创建LoRA模型
        self.model = get_peft_model(self.base_model, self.lora_config)

        # 打印可训练参数数量
        self.model.print_trainable_parameters()

    def prepare_datasets(self, test_size=0.2):
        """准备训练和验证数据集"""
        # 创建完整数据集
        full_dataset = BinaryQADataset(
            self.dataset_path, self.tokenizer
        )

        # 划分训练集和验证集
        dataset_size = len(full_dataset)
        indices = list(range(dataset_size))
        split_idx = int(dataset_size * (1 - test_size))

        train_indices = indices[:split_idx]
        eval_indices = indices[split_idx:]

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)

        print(f"总样本数: {dataset_size}")
        print(f"训练样本数: {len(train_dataset)}")
        print(f"验证样本数: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def train(self, epochs, batch_size, learning_rate=2e-4):  # LoRA通常使用稍大的学习率
        """使用LoRA微调模型"""
        # 准备数据
        train_dataset, eval_dataset = self.prepare_datasets()

        # 训练参数 - 针对LoRA优化
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            warmup_steps=100,  # LoRA训练通常需要较少预热
            weight_decay=0.01,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_steps=50,  # 更频繁的日志记录
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
            gradient_accumulation_steps=2,  # 梯度累积，节省内存
            optim="adamw_torch",  # 使用AdamW优化器
        )

        # 创建Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )

        # 开始训练
        print("开始使用LoRA微调文本匹配模型...")
        trainer.train()

        # 保存最佳模型 - 只保存LoRA权重
        lase_model_dict_path = os.path.join(self.output_dir, "last_model_dict/model_dict.bin")
        lase_model_tokenizer_path = os.path.join(self.output_dir, "last_model_tokenizer")
        lase_model_lora_path = os.path.join(self.output_dir, "last_model_lora")
        trainer.model.save_pretrained(lase_model_lora_path)

        torch.save(trainer.model.state_dict(),lase_model_dict_path)
        self.tokenizer.save_pretrained(lase_model_tokenizer_path)

        print(f"LoRA微调完成，模型权重已保存到: {lase_model_dict_path}")
        return trainer

    def load_lora_model(self, lora_path):
        """加载训练好的LoRA权重"""
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.model.eval()
        print("LoRA权重加载完成")

    def predict(self, question, contexts):
        """预测哪个文本最能回答问题"""
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
                match_score = probs[0][1].item()  # 正类的概率

            scores.append(match_score)

        # 返回排序结果
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
        """合并LoRA权重到基础模型并保存完整模型"""
        # 合并LoRA权重
        merged_model = self.model.merge_and_unload()

        # 保存完整模型
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print(f"完整模型已保存到: {output_path}")
        return merged_model


# 使用示例
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "deepSeek_R1_Distill_Qwen_1.5B")
    # 配置参数
    # model_path = "./DeepSeek_R1_Distill_Qwen_1.5B"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = [os.path.join(current_dir, "dataset/nq_open_train.json"), os.path.join(current_dir, "dataset/nq_open__val.json")]
    # dataset_path = "./datasets/nq_open_train.json"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "text_matching_model")
    # output_dir = "./text_matching_model"

    # 创建训练器
    trainer = TextMatchingTrainer(model_path, dataset_path, output_dir)

    # 开始训练
    trainer.train(epochs=3, batch_size=8, learning_rate=2e-4)



if __name__ == "__main__":
    main()

"""

{'eval_loss': 0.05197696015238762, 'eval_accuracy': 0.9891666666666666, 'eval_f1': 0.9356435643564357, 'eval_precision': 0.9264705882352942, 'eval_recall': 0.945, 'eval_auc': 0.9977590909090908, 'eval_runtime': 79.6044, 'eval_samples_per_second': 30.149, 'eval_steps_per_second': 1.884, 'epoch': 3.0}                                                                                                      
{'train_runtime': 2606.4301, 'train_samples_per_second': 11.05, 'train_steps_per_second': 0.691, 'train_loss': 0.13474373790952895, 'epoch': 3.0}                                                        
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1800/1800 [43:26<00:00,  1.45s/it]
LoRA微调完成！模型已保存到: /tmp/pycharm_project_679/text_matching_model/best_model
LoRA权重文件大小显著小于完整模型

"""