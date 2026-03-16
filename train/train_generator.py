import argparse
import os
from utils import multi_context_dataset
from model.clean_model import CleanModel
from train.generator_model import GeneratorModel
from model.nli_model import NLIModel
from model.text_matching_model import TextMatchingModel
from train.enviroment import RewardCalculator
from ppo_training_wrapper import PPOTrainingWrapper



def __prepare(args: argparse.Namespace):
    dataset = multi_context_dataset(dataset_path=args.dataset_path, context_count=args.context_count, correct_probability=args.correct_probability)
    clean_model = CleanModel(model_path=args.clean_model_path,tokenizer=None, gpu_idx=args.clean_gpu_idx, tokenizer_path=args.clean_model_path)
    text_matching_model = TextMatchingModel(base_model_path=args.matching_base_path, model_dict_path=args.matching_dict_path,
                                            matching_tokenizer_path=args.matching_tokenizer_path, context_clean_model=clean_model,
                                            lora_weight_path=args.matching_lora_weight_path,device_index=args.matching_gpu_idx)
    generator = GeneratorModel(model_path=args.generator_path, gpu_idx=args.generator_gpu_idx, tokenizer_path=args.generator_path)
    nli_model = NLIModel(model_path=args.NLI_model_path, gpu_idx=args.NLI_gpu_idx)
    reward_calculator =  RewardCalculator(matching_model=text_matching_model, nli_model=nli_model)
    print(f"clean:{clean_model.device}, text_matching:{text_matching_model.device}, generator:{generator.device}, nli:{nli_model.device}")
    return dataset, text_matching_model, generator, nli_model, reward_calculator

def main(args: argparse.Namespace):
    dataset, text_matching_model, generator, nli_model, reward_calculator = __prepare(args)
    ppo_training_wrapper = PPOTrainingWrapper(env=reward_calculator, learning_rate=args.lr,
                                              batch_size=args.batch_size, generator=generator,
                                              tokenizer=generator.tokenizer, split_tokens=args.split_tokens,
                                              generator_gpu_idx=args.generator_gpu_idx)
    positive_count = 0
    for epoch in range(args.num_epochs):
        print(f"开始第{epoch + 1}轮训练")
        infos = ppo_training_wrapper.train_episode(training_data=dataset.data_all, epoch=epoch+1)
        print(f"第{epoch + 1}轮统计信息: ", infos["generation_infos"])
        print("总奖励: ", infos["rewards"])
        print("正奖励次数: ", infos["positive_count"])
        if infos["positive_count"] >= positive_count:
            save_path = f"{args.save_path}/{str(args.generator_path).split('/')[-1]}/epoch_{epoch+1}"
            print(f"本次微调之后的正奖励次数{infos['positive_count']}不少于之前次数{positive_count}, 模型保存到：{save_path}")
            os.makedirs(save_path, exist_ok=True)
            ppo_training_wrapper.ppo_trainer.model.save_pretrained(save_path)
            positive_count = infos["positive_count"]


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))[:-6]
    params = argparse.ArgumentParser()
    """
    文本匹配模型
    """
    params.add_argument('--matching_base_path', type=str,
                        help='the path of base model in text matching model',
                        default=os.path.join(current_dir, 'deepSeek_R1_Distill_Qwen_1.5b'))
    params.add_argument('--matching_lora_weight_path', type=str,
                        help='the path of lora weight in text matching model',
                        default=os.path.join(current_dir, 'text_matching_model/last_model_lora'))
    params.add_argument('--matching_dict_path', type=str,
                        help='the path of weight dict in text matching model',
                        default=os.path.join(current_dir, 'text_matching_model/last_model_dict/model_dict.bin'))
    params.add_argument('--matching_tokenizer_path', type=str,
                        help='the path of tokenizer in text matching model',
                        default=os.path.join(current_dir, 'text_matching_model/last_model_tokenizer'))
    """
    数据集
    """
    params.add_argument('--dataset_path', type=str,
                        help="the path of dataset",
                        default=os.path.join(current_dir, 'dataset/nq_open_train.json'))
    params.add_argument('--context_count', type=int,
                        help="the count of the context for each question",
                        default=10)
    params.add_argument('--correct_probability', type=float,
                        help="the probability of what question contains right context in all question",
                        default=0.8)
    """
    clean模型（开源模型即可）
    """
    params.add_argument('--clean_model_path', type=str,
                        help='the path of clean model',
                        default=os.path.join(current_dir, 'llama3.2_3b_instruct'))
    """
    NLI模型（开源模型即可）
    """
    params.add_argument('--NLI_model_path', type=str,
                        help='the path of NLI model',
                        default=os.path.join(current_dir, 'tasksource_deberta_small_long_nli'))
    """
    生成器模型
    """
    params.add_argument('--generator_path', type=str,
                        help='the path of generator, pretrain model or fine-turned model.',
                        default=os.path.join(current_dir, 'llama3.2_3b_instruct'))
    params.add_argument('--generator_lora_path', type=str,
                        help='the path of lora weight in fine-turned model.',
                        default="")
    params.add_argument('--split_tokens', nargs='+',
                        help='the split_tokens, when use thinking model or need instruction answer.',
                        default=["</think>", "<Answer>:"])
    """
    GPU分配
    """
    params.add_argument('--matching_gpu_idx', type=int, default=2)
    params.add_argument('--clean_gpu_idx', type=int, default=4)
    params.add_argument('--NLI_gpu_idx', type=int, default=3)
    params.add_argument('--generator_gpu_idx', type=int, default=0)

    """
    训练超参数
    """
    params.add_argument('--num_epochs', type=int, default=20)
    params.add_argument('--lr', type=float, default=1e-6)
    params.add_argument('--batch_size', type=int, default=16)

    """
    保存路径
    """
    params.add_argument('--save_path', type=str, default=os.path.join(current_dir, "generator"))

    args = params.parse_args()
    main(args)