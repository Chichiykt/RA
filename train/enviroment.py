from typing import List

from model.text_matching_model import TextMatchingModel
from model.nli_model import NLIModel

class RewardCalculator:
    """Reinforcement Learning Environment Demo"""

    def __init__(self, matching_model: TextMatchingModel, nli_model: NLIModel):

        self.matching_model = matching_model
        self.nli_model = nli_model

        self.generate_info = {
            "A1": 0, # Can be answered + Correct answer + len < 25
            "A2": 0, # Can be answered + Correct answer + len >= 25
            "B1": 0, # Do not answer + “I don’t know” + len < 25
            "B2": 0, # Do not answer + “I don’t know” + len > 25
            "C1": 0, # Do not answer + “internal knowledge” + len < 25
            "C2": 0, # Do not answer + “internal knowledge” + len > 25
            "D1": 0, # You can reply with "I don't know" + len < 25
            "D2": 0, # You can reply with "I don't know" + len > 25
            "E1": 0, # Can be answered + Incorrect answer + len < 25
            "E2": 0, # Can be answered + Incorrect answer + len > 25
            "F1": 0, # Cannot answer + Incorrect answer + len < 25
            "F2": 0,  # Cannot answer + Incorrect answer + len > 25
            "E": 0, # Duplicate answer
        }

    def generate_info_init(self):
        self.generate_info = {
            "A1": 0, 
            "A2": 0, 
            "B1": 0, 
            "B2": 0,
            "C1": 0, 
            "C2": 0,
            "D1": 0, 
            "D2": 0, 
            "E1": 0,
            "E2": 0,
            "F1": 0,
            "F2": 0, 
            "E": 0,
        }

    """
        Determine whether a response is possible based on the text-matching model
    """
    def check_answerable(self, question: str, reference_texts: List[str]) -> bool:
        """Check whether the question can be answered"""
        for ref_text in reference_texts:
            if self.matching_model.predict(question, ref_text):
                print("Question：",question ,"Can be answered")
                return True
        print("Question：", question , "Cannot be answered")
        return False

    """
        Check whether the answer is correct (based on the documentation, or if the answer is ‘I don’t know’)
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

        if answer.count('\n') > 1 or answer.count(" ") > 50: # Repeated thoughts, repeated answers
            print("token limit")
            return "error"

        if "entailment" == self.nli_model.predict(unknown_template, answer)[0]:
            print(f"--------------------------------------------------Classified as 'Unanswerable.'")
            return "unknown"

        for key_word in target_answer_words:
            if key_word.lower() not in answer.lower():
                key_correct = False
                break

        if "entailment" == self.nli_model.predict(correct_reference, answer)[0]:
            if not answerable and key_correct:
                print(f"--------------------------------------------------(Internal knowledge) Determined to be an ‘implication’, the model’s answer is correct")
            else:
                if key_correct:
                    print(f"--------------------------------------------------Determined to be an ‘implication’, the model’s answer is correct")
                else:
                    print(f"---------------------------------------NLI misclassification")
                    return "incorrect"
            return "correct"
        else:
            if not answerable:
                print(f"--------------------------------------------------(I don’t know where it came from) marked as ‘incorrect’")
            else:
                print(f"--------------------------------------------------Marked as 'incorrect'")
            return "incorrect"

    """
        Can be answered + Correct answer = +  
        Can be answered + Incorrect answer = -  
        Can be answered + "Cannot answer" = -   
        Cannot be answered + "Cannot answer" = +     
        Cannot be answered + "Incorrect answer" = -    
        Cannot be answered + "Correct answer" (internal knowledge) = +
    """
    def calculate_reward(self, answerable: bool, answer_result: str, answer_len: int, episode:int) -> float:
        if answer_result == "error":
            if episode == 0:
                self.generate_info["E"] += 1
                return -2.0
            else:
                self.generate_info["E"] += 1
                return -2.0


        """Calculate the reward value"""
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
