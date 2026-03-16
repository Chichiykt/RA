import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class CleanModel:
    def __init__(self, model_path, tokenizer, gpu_idx, tokenizer_path=None):
        self.device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                          torch_dtype=torch.float16
                                                          ).to(self.device)
        if tokenizer_path is None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"clean_model在{self.model.device}上加载完毕")
    def clean_context(self, context):
        clean_template = [
            {"role": "user", "content": f"""
            Please distill complete, accurate, and coherent information from the user-provided text and output it as a single, smooth, concise declarative paragraph. Follow these principles:

                Ignore distractions: Directly skip any obviously incomplete, truncated, or irrelevant sentence fragments at the beginning, end, or within the text.
            
                Extract Essentials: Focus solely on extracting sentences that are relatively complete and contain explicit facts (such as events, data, timelines, or relationships).
            
                Integrate and Cohere: Naturally merge all extracted key information into a coherent paragraph based on logical structure (e.g., by theme, timeline, or category). Ensure sentences flow smoothly with natural transitions.
            
                Factual Accuracy: Use only data and facts provided in the original text. Do not add any information not present in the source material or make subjective assumptions.
                
                Explicit reference: When encountering pronouns (such as they, it, their, etc.), infer and explicitly identify the subject they refer to based on context (e.g., a country, team, or organization), and directly use that subject's name in the output to ensure paragraph clarity.
                    
                Extract Core: Focus solely on extracting sentences that are relatively complete and contain explicit facts. When encountering pronouns, determine their antecedents (e.g., countries, teams) based on contextual logic, and use the antecedent's name directly when integrating information.
                
                Place your reply after <Answer>: .
                Example:
                    Text:
                        tied and 1 no result match as well. They have won the 2014 ICC World Twenty20 championship in Bangladesh and was runner-up in two previous occasions. (2009, 2012). As of June 2018, Sri Lanka have faced all nine teams in Test cricket, with their most frequent opponent being Pakistan, playing 53 matches against them. Sri Lanka have registered more wins against Pakistan and Bangladesh than any other team, with 16. In ODI matches, Sri Lanka have played against 17 teams; they have played against India most frequently, with a winning percentage of 39.49 in 158 matches. Within usual major ODI
                    Your response: 
                        <Answer>:The Sri Lanka cricket team won the ICC World Twenty20 Championship in Bangladesh in 2014 and had previously finished as runners-up twice in 2009 and 2012. As of June 2018, Sri Lanka had played all nine Test-playing nations, with Pakistan being their most frequent opponent at 53 matches. They have secured the most victories against Pakistan and Bangladesh, each with 16 wins. In One-Day Internationals (ODIs), Sri Lanka has played 17 teams, with India being the most frequent opponent. The two sides have contested 158 matches, where Sri Lanka holds a 39.49% win rate.
                This task:
                    Text:f{context}
                """

             }
        ]
        return self.get_answer(clean_template)

    def get_answer(self, template):
        clean_template = self.tokenizer.apply_chat_template(
            template,
            tokenize=False,
            add_generation_prompt=True
        )

        clean_inputs = self.tokenizer(
            clean_template,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            clean_inputs.to(self.device)
            clean_outputs = self.model.generate(
                input_ids=clean_inputs["input_ids"],
                attention_mask=clean_inputs['attention_mask'],
                temperature=0.6,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(clean_outputs[0], skip_special_tokens=True).split("<Answer>:")[-1].strip()

    def question_answer_combine(self, question, answer:list):
        combine_template = [
            {"role": "user", "content": f"""
                Your task is to merge the given question statement and answer statement into a single sentence, placing your final answer after <Answer>:.
                Below are some reference examples:
                Example 1:
                Question: Which country does Chongqing belong to? Answer: [China].
                Your response: <Answer>: Chongqing belongs to China.
                Example 2:
                Question: In which city were the 2008 Olympic Games held? Answer: [Beijing].
                Your response: <Answer>: The 2008 Summer Olympics were held in Beijing.
                Example 3: Question: Which singer composed the famous song “Late”? Answer: [Xue Zhiqian,zhouyili].
                Your response: <Answer>: “Late” was composed by Xue Zhiqian and zhouyili.
                This task:
                    Question:{question} Answer:{str(answer)}
                """
             }
        ]
        return self.get_answer(combine_template)
