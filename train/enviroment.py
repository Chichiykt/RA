from typing import List

from model.text_matching_model import TextMatchingModel
from model.nli_model import NLIModel

class RewardCalculator:
    """强化学习环境"""

    def __init__(self, matching_model: TextMatchingModel, nli_model: NLIModel):

        self.matching_model = matching_model
        self.nli_model = nli_model

        self.generate_info = {
            "A1": 0, # 可回答 + 回答正确 + len < 25
            "A2": 0, # 可回答 + 回答正确 + len >= 25
            "B1": 0, # 不可回答 + “我不知道” + len < 25
            "B2": 0, # 不可回答 + “我不知道” + len > 25
            "C1": 0, # 不可回答 + “内部知识” + len < 25
            "C2": 0, # 不可回答 + “内部知识” + len > 25
            "D1": 0, # 可回答 + "我不知道" + len < 25
            "D2": 0, # 可回答 + "我不知道" + len > 25
            "E1": 0, # 可回答 + 回答错误 + len < 25
            "E2": 0, # 可回答 + 回答错误 + len > 25
            "F1": 0, # 不可回答 + 回答错误 + len < 25
            "F2": 0,  # 不可回答 + 回答错误 + len > 25
            "E": 0, # 重复答案
        }

    def generate_info_init(self):
        self.generate_info = {
            "A1": 0,  # 可回答 + 回答正确 + len < 25
            "A2": 0,  # 可回答 + 回答正确 + len >= 25
            "B1": 0,  # 不可回答 + “我不知道” + len < 25
            "B2": 0,  # 不可回答 + “我不知道” + len > 25
            "C1": 0,  # 不可回答 + “内部知识” + len < 25
            "C2": 0,  # 不可回答 + “内部知识” + len > 25
            "D1": 0,  # 可回答 + "我不知道" + len < 25
            "D2": 0,  # 可回答 + "我不知道" + len > 25
            "E1": 0,  # 可回答 + 回答错误 + len < 25
            "E2": 0,  # 可回答 + 回答错误 + len > 25
            "F1": 0,  # 不可回答 + 回答错误 + len < 25
            "F2": 0,  # 不可回答 + 回答错误 + len > 25
            "E": 0,  # 重复答案
        }

    """
        根据文本匹配模型判断是否可回答
    """
    def check_answerable(self, question: str, reference_texts: List[str]) -> bool:
        """检查问题是否可回答"""
        for ref_text in reference_texts:
            if self.matching_model.predict(question, ref_text):
                print("问题：",question ,"可回答")
                return True
        print("问题：", question , "不可回答")
        return False

    """
        检查回答是否正确（基于文档回答或者回答我不知道）
    """
    def check_answer_correctness(self, answer: str, correct_reference, answerable, target_answer_words) -> str:
        unknown_template = """An effective response must provide substantive information relevant to the question.
         If the response indicates that the model is unable to provide an answer due to knowledge limitations, capability constraints, or strategic reasons, it is considered invalid. 
        Such responses typically include (but are not limited to) the following expressions: 
        “I don't know,” “I don't have the relevant information,” “I cannot answer,” “As an AI model...,” “Based on my knowledge base...,” 
        or evading the question with irrelevant content."""
        print("check_answer_correctness(), A:",  answer)
        print("target_answer_words: ", target_answer_words)

        key_correct = True

        if answer.count('\n') > 1 or answer.count(" ") > 50: # 重复思考、重复答案
            print("啰嗦")
            return "error"

        if "entailment" == self.nli_model.predict(unknown_template, answer)[0]:
            print(f"--------------------------------------------------判定为“无法回答”")
            return "unknown"

        for key_word in target_answer_words:
            if key_word.lower() not in answer.lower():
                key_correct = False
                break

        if "entailment" == self.nli_model.predict(correct_reference, answer)[0]:
            if not answerable and key_correct:
                print(f"--------------------------------------------------(内部知识)判定为“蕴含关系”, 模型回答正确")
            else:
                if key_correct:
                    print(f"--------------------------------------------------判定为“蕴含关系”, 模型回答正确")
                else:
                    print(f"---------------------------------------NLI判断失误")
                    return "incorrect"
            return "correct"
        else:
            if not answerable:
                print(f"--------------------------------------------------(不知哪来的)判定为“回答错误”")
            else:
                print(f"--------------------------------------------------判定为“回答错误”")
            return "incorrect"

    """
        可回答+回答正确 = +
        可回答+回答错误 = -
        可回答+"无法回答" = -
        不可回答+"无法回答" = +
        不可回答+"回答错误" = -
        不可回答+"回答正确"（内部知识） = +
    """
    def calculate_reward(self, answerable: bool, answer_result: str, answer_len: int, episode:int) -> float:
        if answer_result == "error":
            if episode == 0:
                self.generate_info["E"] += 1
                return -2.0
            else:
                self.generate_info["E"] += 1
                return -2.0


        """计算奖励值"""
        if answerable and answer_result == "correct":
            if episode == 0:
                if answer_len < 25:
                    self.generate_info["A1"] += 1
                    return 1.0
                else:
                    self.generate_info["A2"] += 1
                    return 0.7
                # return 1.0
            elif episode == 1:
                if answer_len < 25:
                    self.generate_info["A1"] += 1
                    return 0.6
                else:
                    self.generate_info["A2"] += 1
                    return 0.3
            else:
                if answer_len < 25:
                    self.generate_info["A1"] += 1

                else:
                    self.generate_info["A2"] += 1
                return 0.4

        elif answerable and answer_result == "incorrect":
            if episode == 0:
                if answer_len > 25:
                    self.generate_info["E2"] += 1
                    return -1.0
                else:
                    self.generate_info["E1"] += 1
                    return -0.7
                # return -1.5
            elif episode == 1:
                if answer_len > 25:
                    self.generate_info["E2"] += 1
                    return -1.0
                else:
                    self.generate_info["E1"] += 1
                    return -0.7
            else:
                if answer_len > 25:
                    self.generate_info["E2"] += 1
                    return -1.0
                else:
                    self.generate_info["E1"] += 1
                    return -0.7
                # return 0.0
        elif answerable and answer_result == "unknown":
            if episode == 0:
                if answer_len < 25:
                    self.generate_info["D1"] += 1
                    return -0.7
                else:
                    self.generate_info["D2"] += 1
                    return -1.0
                # return -1.2
            elif episode == 1:
                if answer_len < 25:
                    self.generate_info["D1"] += 1
                    return -0.7
                else:
                    self.generate_info["D2"] += 1
                    return -1.0
            else:
                if answer_len < 25:
                    self.generate_info["D1"] += 1
                    return -0.7
                else:
                    self.generate_info["D2"] += 1
                    return -1.0
                # return 0.0
        elif not answerable and answer_result == "unknown":
            if episode == 0:
                if answer_len < 25:
                    self.generate_info["B1"] += 1
                    return 1.0
                else:
                    self.generate_info["B2"] += 1
                    return 0.7
                # return 1.0
            elif episode == 1:
                if answer_len < 25:
                    self.generate_info["B1"] += 1
                    return 0.6
                else:
                    self.generate_info["B2"] += 1
                    return 0.3
            else:
                if answer_len < 25:
                    self.generate_info["B1"] += 1
                    return 0.6
                else:
                    self.generate_info["B2"] += 1
                    return 0.3
                # return 0.4
        elif not answerable and answer_result == "incorrect":
            if episode == 0:
                if answer_len < 25:
                    self.generate_info["F1"] += 1
                    return -0.7
                else:
                    self.generate_info["F2"] += 1
                    return -1.0
                # return -1.2
            elif episode == 1:
                if answer_len < 25:
                    self.generate_info["F1"] += 1
                    return -0.7
                else:
                    self.generate_info["F2"] += 1
                    return -1.0
                # return -1.2
            else:
                if answer_len < 25:
                    self.generate_info["F1"] += 1
                    return -0.7
                else:
                    self.generate_info["F2"] += 1
                    return -1.0
                # return 0.0
        elif not answerable and answer_result == "correct":
            if episode == 0:
                if answer_len < 25:
                    self.generate_info["C1"] += 1
                    return 1.0
                else:
                    self.generate_info["C2"] += 1
                    return 0.7
                # return 1.0
            elif episode == 0:
                if answer_len < 25:
                    self.generate_info["C1"] += 1
                    return 1.0
                else:
                    self.generate_info["C2"] += 1
                    return 0.7
                # return 1.2
            else:
                if answer_len < 25:
                    self.generate_info["C1"] += 1
                    return 1.0
                else:
                    self.generate_info["C2"] += 1
                    return 0.7
                # return 0.5
        # return 0.0
