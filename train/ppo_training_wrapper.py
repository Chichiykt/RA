import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from trl import PreTrainedModelWrapper, PPOConfig, PPOTrainer

from train.enviroment import RewardCalculator


class PPOTrainingWrapper:
    def __init__(self, env: RewardCalculator, learning_rate: float, batch_size: int, generator, tokenizer: PreTrainedTokenizerBase, split_tokens:list, generator_gpu_idx:int):
        self.env = env
        self.batch_size = batch_size
        self.split_tokens = split_tokens
        # PPO配置
        self.ppo_config = PPOConfig(
            model_name="",
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            ppo_epochs=4,
            log_with=None,
            use_score_scaling=False,
            use_score_norm=False,
            whiten_rewards=False,
            cliprange=0.1,
            cliprange_value=0.1,
            vf_coef=0.3,
            gamma=0.99,
            lam=0.95,
            target_kl=6.0,
            init_kl_coef=0.2,
            adap_kl_ctrl=True,


        )
        self.generator_gpu_idx = generator_gpu_idx
        self.generator = generator
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=generator.model,
            ref_model=None,  # 使用原始模型作为参考
            tokenizer=tokenizer,

        )

    def train_episode(self, training_data: list[dict], epoch:int):
        """训练一个epoch"""
        infos = {
                "rewards":0,
                "positive_count":0,
                "generation_infos" : []
            }
        for i in range(0, len(training_data), self.batch_size):
            batch_data = training_data[i:i + self.batch_size]


            questions = [data["question"] for data in batch_data]
            reference_texts_list = [data["contexts"] for data in batch_data]
            correct_references = [data["correct_context"] for data in batch_data]
            answer_list = [data["answer"] for data in batch_data]


            query_tensors = []
            response_tensors = []
            attention_masks = []
            answers = []
            rewards = []
            for question, reference_texts, correct_reference, target_answer_words in zip(questions, reference_texts_list, correct_references, answer_list):
                inputs_tensor = self.generator.get_input_ids(question, reference_texts)
                query_tensors.append(inputs_tensor)
                response = self.ppo_trainer.generate(query_tensor=inputs_tensor,
                                                     return_prompt=False,
                                                     max_new_tokens=4096,
                                                     temperature=0.6
                                                     )
                response_tensors.append(response[0])
                answer_str = self.ppo_trainer.tokenizer.decode(response[0], skip_special_tokens=True).split(self.split_tokens[0])[-1].strip().split(self.split_tokens[1])[-1].strip()
                answers.append((answer_str, inputs_tensor, response[0]))

            # 计算奖励
            #     answerable = self.env.check_answerable(question, reference_texts)
                answerable = correct_reference in reference_texts
                print(question,"可回答" if answerable else "不可回答")
                rewards.append(torch.FloatTensor(torch.tensor(self.env.calculate_reward(
                    answerable,
                    self.env.check_answer_correctness(answer_str, correct_reference, answerable, target_answer_words),
                    str(answer_str).count(' ') + 1,
                    epoch
                ))))

            self.ppo_trainer.step(
                queries=query_tensors,
                responses=response_tensors,
                response_masks=None,
                scores=rewards
            )

            print(f"Batch {i // self.batch_size + 1}/{(len(training_data) - 1) // self.batch_size + 1}")
            print(f"Average reward: {np.mean(rewards):.4f}")
            for j, (question, answer, reward) in enumerate(zip(questions, answers, rewards)):
                print(f"  Q: {question[:50]}...")
                print(f"  A: {answer[0][:50]}...")
                print(f"  Reward: {reward:.4f}")
            print("-" * 50)
            infos["rewards"] += np.sum(rewards)
            infos["positive_count"] += len(list(filter(lambda x: x > 0, rewards)))
            print("统计信息：", self.env.generate_info)

            infos["generation_infos"].append({f"episode_{epoch}_{i + 1}": self.env.generate_info.copy()})
            self.env.generate_info_init()
        infos["generation_infos"].append({
                key: sum(list(info.values())[0][key] for info in infos["generation_infos"])
                for key in self.env.generate_info.keys()})
        return infos