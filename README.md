# RA
code for paper [RA: Combating Hallucinations in RAG-Based LLMs by Reinforcing Attention on Critical Texts]

requires:  
  python>=3.10.0  
  datasets>=2.18.0  
  numpy>=1.26.4  
  pandas>=2.3.3  
  peft==0.18.0  
  pydantic==2.12.5  
  sklearn  
  torch==2.6.0  
  tqdm==4.67.1  
  transformers==4.52.0  
  trl==0.8.1  
  
# All main training programmes involve complex dependencies; please ensure that all dependencies are of the correct version and that all code files are complete, as failure to do so may result in unforeseen errors.
We have defined a number of custom classes, including custom dataset loading classes, utility classes, rudimentary reinforcement learning feedback environment classes, and base classes for various models. These classes are interdependent throughout the project’s operation; whilst they do not fully embody the principles of high cohesion and low coupling, they function without issue. Please contact us should you encounter any problems. Below, we will outline the two main training entry points.

When you wish to train a text-matching model or any tool model from scratch, please refer to the implementation in ft_text_matching.py. Before doing so, ensure that the tRL and Transformers dependencies are properly configured. When loading the base model, you can adapt it to your task type by modifying the `num_labels` and `problem_type` parameters. Adjust the LORA configuration options as required. The base model path, custom dataset class and custom dataset path can all be modified within the code. 
You can then use it directly. python ft_text_matching.py Run the code. You can also modify the code logic to suit your needs.

ft_text_matching.py：Specify a base model and use LORA technology to perform efficient fine-tuning of text-matching models.

If you wish to reproduce the training of the generator model using the PPO algorithm and our method, please refer to `train_generator.py`. We have encapsulated all the hyperparameters, so you can simply modify them to suit your task.
train_generator.py: Train the generative model

# Our custom data is stored in the ‘dataset’ folder, which contains a total of three datasets.
For other datasets used in the paper, please refer to the text; here we are only publishing the datasets we have created ourselves.
