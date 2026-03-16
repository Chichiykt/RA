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

# 所有的主训练程序都包含着复杂的依赖关系，请务必保证所有的依赖版本正确和代码文件完整，否则会引发不可预知的错误。
我们定义了很多的自定义类，包括自定义的数据集加载类、工具类、粗略的强化学习反馈环境类、各种模型的主类等，他们在整个项目的运行中都相互依赖，
虽未能体现出高内聚低耦合的特性，但是运行是没有问题的。如果有任何问题请联系我们。下面我们主要介绍两个主要的训练入口。


当你想要从0开始训练文本匹配模型或者任何工具模型的时候，请参考ft_text_matching.py的实现。在此之前务必配置好trl和transformers依赖。在加载基座模型的时候，
可以通过更改num_lables参数和problem_type参数来适配你的任务类型。lora配置项中根据你的需要进行更改。基座模型路径、自定义数据集类、自定义数据集路径都在代码中更改。
随后直接使用 
##python ft_text_matching.py
运行代码。你也可以根据你的需要更改代码逻辑。
##ft_text_matching.py：指定基座模型，使用lora技术进行参数高效微调文本匹配模型。

当你想要复现使用PPO算法和我们的方法训练生成器模型，请关注train_generator.py，我们已经将所有的超参数封装好，你可以直接修改以适配你的任务。
train_generator.py: 进行生成器模型的训练

# 我们自定义的数据集体现在dataset文件夹中，一共包含三个数据集。
文章中使用的其他数据集请参照文章表述，这里我们只公布我们自主创建的数据集。
