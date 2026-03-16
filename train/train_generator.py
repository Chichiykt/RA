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
        print(f"Start training round {epoch + 1}")
        infos = ppo_training_wrapper.train_episode(training_data=dataset.data_all, epoch=epoch+1)
        print(f"Statistics for epoch {epoch + 1}: ", infos["generation_infos"])
        print("Total prize money: ", infos["rewards"])
        print("Number of positive rewards: ", infos["positive_count"])
        if infos["positive_count"] >= positive_count:
            save_path = f"{args.save_path}/{str(args.generator_path).split('/')[-1]}/epoch_{epoch+1}"
            print(f"Following this fine-tuning, the number of positive rewards {infos['positive_count']} is no less than the previous count {positive_count}. The model has been saved to: {save_path}")
            os.makedirs(save_path, exist_ok=True)
            ppo_training_wrapper.ppo_trainer.model.save_pretrained(save_path)
            positive_count = infos["positive_count"]


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))[:-6]
    params = argparse.ArgumentParser()
    """
    Text matching model
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
    Dataset
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
    clean model (an open-source model will suffice)
    """
    params.add_argument('--clean_model_path', type=str,
                        help='the path of clean model',
                        default=os.path.join(current_dir, 'llama3.2_3b_instruct'))
    """
    NLI model (an open-source model will suffice)
    """
    params.add_argument('--NLI_model_path', type=str,
                        help='the path of NLI model',
                        default=os.path.join(current_dir, 'tasksource_deberta_small_long_nli'))
    """
    Generative models
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
    GPU allocation
    """
    params.add_argument('--matching_gpu_idx', type=int, default=2)
    params.add_argument('--clean_gpu_idx', type=int, default=4)
    params.add_argument('--NLI_gpu_idx', type=int, default=3)
    params.add_argument('--generator_gpu_idx', type=int, default=0)

    """
    Training hyperparameters
    """
    params.add_argument('--num_epochs', type=int, default=20)
    params.add_argument('--lr', type=float, default=1e-6)
    params.add_argument('--batch_size', type=int, default=16)

    """
    Save path
    """
    params.add_argument('--save_path', type=str, default=os.path.join(current_dir, "generator"))

    args = params.parse_args()
    main(args)
