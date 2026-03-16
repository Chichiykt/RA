import argparse
import os
from model.nli_model import NLIModel
from utils import multi_context_dataset
from model.text_matching_model import TextMatchingModel
from model.clean_model import CleanModel

from model.generator_model import Generator
from train.enviroment import RewardCalculator

def __prepare(args: argparse.Namespace):
    dataset = multi_context_dataset(dataset_path=args.dataset_path, context_count=args.context_count, correct_probability=args.correct_probability)
    clean_model = CleanModel(model_path=args.clean_model_path,tokenizer=None, gpu_idx=args.clean_gpu_idx, tokenizer_path=args.clean_model_path)
    text_matching_model = TextMatchingModel(base_model_path=args.matching_base_path, model_dict_path=args.matching_dict_path,
                                            matching_tokenizer_path=args.matching_tokenizer_path, context_clean_model=clean_model,
                                            lora_weight_path=args.matching_lora_weight_path,device_index=args.matching_gpu_idx)
    generator = Generator(base_model_path=args.generator_path, tokenizer_path=args.generator_path,gpu_idx=args.generator_gpu_idx, lora_path=args.generator_lora_path)
    nli_model = NLIModel(model_path=args.NLI_model_path, gpu_idx=args.NLI_gpu_idx)
    reward_calculator =  RewardCalculator(matching_model=text_matching_model, nli_model=nli_model)
    return dataset, text_matching_model, generator, nli_model, reward_calculator
def main(args: argparse.Namespace):
    dataset, text_matching_model, generator, nli_model, reward_calculator = __prepare(args)
    for epoch in range(args.num_epochs):
        for singe_data in dataset.data_all:
            answerable = singe_data["correct_context"] in singe_data["contexts"]
            print(singe_data["question"],"Can answer" if answerable else "Do not answer")
            answer = generator.generate(question=singe_data["question"], reference_list=singe_data["contexts"], split_tokens=args.split_tokens)
            answer_result = reward_calculator.check_answer_correctness(answer=answer, correct_reference=singe_data["correct_context"],
                                                                       answerable=answerable, target_answer_words=singe_data["answer"])
            reward_calculator.calculate_reward(answerable=answerable, answer_result=answer_result, answer_len=len(answer), episode=epoch)
        print(f"----epoch_{epoch+1}: {reward_calculator.generate_info}")
        reward_calculator.generate_info_init()


if __name__ == '__main__':

    current_dir = os.path.dirname(os.path.abspath(__file__))[:-5]
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
    """·
    Generative models
    """
    params.add_argument('--generator_path', type=str,
                        help='the path of generator, pretrain model or fine-turned model.',
                        default=os.path.join(current_dir, 'llama3_8b_instruct'))
    params.add_argument('--generator_lora_path', type=str,
                        help='the path of lora weight in fine-turned model.',
                        # default=os.path.join(current_dir,"generator/best_models/qwen3_8b/epoch_19"))
                        default=None)
    params.add_argument('--split_tokens', nargs='+',
                        help='the split_tokens, when use thinking model or need instruction answer.',
                        default=["</think>","<Answer>:"])
    """
    GPU allocation
    """
    params.add_argument('--matching_gpu_idx', type=int, default=2)
    params.add_argument('--clean_gpu_idx', type=int, default=5)
    params.add_argument('--NLI_gpu_idx', type=int, default=1)
    params.add_argument('--generator_gpu_idx', type=int, default=6)

    """
    Evaluation round
    """
    params.add_argument('--num_epochs', type=int, default=3)

    args = params.parse_args()
    main(args)
