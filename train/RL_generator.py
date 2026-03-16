import os
from typing import List, Dict

import numpy as np
import torch
from trl import PPOConfig, PPOTrainer

from enviroment import RewardCalculator
from generator_model import GeneratorModel
from multi_context_dataset import MultiContextDataset
# from log_function import logger
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
class PPOTrainingWrapper:
    def __init__(self, env: RewardCalculator,
                 learning_rate: float = 1e-6, batch_size: int = 16):
        self.env = env
        self.batch_size = batch_size

        # PPO配置
        self.ppo_config = PPOConfig(
            model_name="RL_deepseek-r1-7b",
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
            adap_kl_ctrl=True

        )
        # logger.info("ppozhhdia1", next(env.generator.model.parameters()).device)
        # 初始化PPOTrainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,

            model=env.generator.model,
            ref_model=None,  # 使用原始模型作为参考
            tokenizer=env.generator.tokenizer,
            # d={}
        )

        # logger.info("ppozhhdia2",next(env.generator.model.parameters()).device)
    def train_episode(self, training_data: List[Dict], episode:int):
        """训练一个周期"""
        all_stats = []
        infos = {
                "rewards":0
            }

        # 分批处理训练数据
        for i in range(0, len(training_data), self.batch_size):
            batch_data = training_data[i:i + self.batch_size]

            # 准备输入数据
            prompts = [data["prompt"] for data in batch_data]
            questions = [data["question"] for data in batch_data]
            reference_texts_list = [data["reference_texts"] for data in batch_data]
            correct_references = [data["correct_context"] for data in batch_data]
            query_tensors = []
            response_tensors = []
            attention_masks = []
            answers = []
            rewards = []
            for prompt, question, reference_texts, correct_reference in zip(prompts, questions, reference_texts_list, correct_references):


                inputs_tensor = self.env.generator.generate_answer_by_reference_texts(
                    prompt, question, reference_texts
                )
                # logger.info("answer-----:", answer_str, "\n\ninputs----:", inputs_tensor, "\n\noutputs------:", outputs_tensor)

                query_tensors.append(inputs_tensor)

                response = self.ppo_trainer.generate(query_tensor=inputs_tensor,
                                                     return_prompt=False,
                                                     # batch_size=1,
                                                     max_new_tokens=4096,
                                                     do_sample=True,
                                                     temperature=0.6,
                                                     # return_dict_in_generate=True,
                                                     # output_attentions=False,
                                                     # output_hidden_states=False,
                                                     pad_token_id=self.env.generator.tokenizer.pad_token_id,
                                                     eos_token_id=self.env.generator.tokenizer.eos_token_id,
                                                     )
                attention_mask = torch.ones_like(response[0], dtype=torch.long)
                response_tensors.append(response[0])
                attention_masks.append(attention_mask)
                answer_str = self.env.generator.tokenizer.decode(response[0], skip_special_tokens=True).split("</think>")[-1].strip()
                # print("模型回复：-------" + answer_str)

                answers.append((answer_str, inputs_tensor, response[0]))

            # 计算奖励
                answerable = self.env.check_answerable(question, reference_texts)
                rewards.append(torch.FloatTensor(torch.tensor(self.env.calculate_reward(
                    answerable,
                    self.env.check_answer_correctness(answer_str, reference_texts, correct_reference, answerable),
                    str(answer_str).count(' ') + 1,
                    episode
                ))))

            # PPO训练
            stats = self.ppo_trainer.step(
                queries=query_tensors,
                responses=response_tensors,
                response_masks=attention_masks,
                scores=rewards
            )


            all_stats.append(stats)

            # 打印进度
            print(f"Batch {i // self.batch_size + 1}/{(len(training_data) - 1) // self.batch_size + 1}")
            print(f"Average reward: {np.mean(rewards):.4f}")
            for j, (question, answer, reward) in enumerate(zip(questions, answers, rewards)):
                print(f"  Q: {question[:50]}...")
                print(f"  A: {answer[0][:50]}...")
                print(f"  Reward: {reward:.4f}")
            print("-" * 50)
            infos["rewards"] += np.sum(rewards)
        return all_stats, infos

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))[:-5]
    generator_path = os.path.join(current_dir, "deepSeek_R1_Distill_Qwen_7B")

    match_model_path = os.path.join(current_dir, "deepSeek_R1_Distill_Qwen_1.5B")
    lora_weight_path = os.path.join(current_dir, "text_matching_model/best_model")
    nli_model_path = os.path.join(current_dir, "tasksource_deberta_small_long_nli")
    dataset_path = os.path.join(current_dir, "dataset/nq_open_train.json")

    save_path = os.path.join(current_dir, "save_model")
    sing_count = 64

    # 初始化组件
    generator = GeneratorModel(generator_path)
    env = ReinforcementLearningEnvironment(
        generator=generator,
        matching_model_path=match_model_path,
        lora_weight_path=lora_weight_path,
        nli_model_path=nli_model_path
    )

    # 初始化PPO训练器
    ppo_wrapper = PPOTrainingWrapper(env)

    multi_context_dataset = MultiContextDataset(dataset_path=dataset_path, count=50, tokenizer=None)

    best_avg_reward = 0
    rewards = 0
    generation_infos = []
    for i in range(20): # 在数据集上学习
        # 开始训练

        for episode in range(15):  # 单个数据集

            print(f"开始第 {episode + 1} 周期训练")
            stats, infos = ppo_wrapper.train_episode(multi_context_dataset.data_all[episode * sing_count : (episode + 1) * sing_count], i)
            print(f"第 {episode + 1} 周期训练完成，平均奖励: {infos['rewards'] / sing_count}")
            rewards += infos['rewards']
            generation_infos.append({f"episode_{i+1}_{episode+1}": env.generate_info.copy()})
            print(f"episode_{i+1}_{episode+1}----------------------", env.generate_info.copy())
            env.generate_info_init()
        epo_info_tmp = {key: sum(list(info.values())[0][key] for info in generation_infos[i*15 + i:(i+1)*15 + i])
                                 for key in env.generate_info.keys()}
        generation_infos.append({f"epoch_{i+1}":epo_info_tmp})
        print(f"epoch_{i+1}-----------------",epo_info_tmp)

        if i == 0:
            ppo_wrapper.ppo_trainer.save_pretrained(save_directory=save_path)
            best_avg_reward = rewards / 15
            print(f"首轮平均奖励{best_avg_reward}")
            rewards = 0
        else:
            avg_reward = rewards / 15
            print(f"第{i+1}轮平均奖励{avg_reward}")
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                print(f"保存到best_model")
                ppo_wrapper.ppo_trainer.save_pretrained(save_directory=save_path)
            else:
                print(f"保存到epoch{i+1}_model")
                ppo_wrapper.ppo_trainer.save_pretrained(save_directory=f"{save_path}/epoch{i+1}_model")
        print("训练完毕")

        print(f"best_avg_reward: {best_avg_reward}")

        print(generation_infos)
if __name__ == "__main__":
    main()
"""


"""
