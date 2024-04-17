# Bert

GPT 的竞品

<<Attention is all you need>>


dimension parametes: feature??

encoder/decoder

input -> 词句的单词的

# problem
1. 要做的事是什么，具体来说？
2. 我目前需要学习哪些东西？
3. 我们要做大模型学习，那么具体用哪个大模型来，看起来像是用bert学习
4. 币圈问题


预测虚拟市场会发生的事件
抵押虚拟货币的货币
流动性的问题

当用100w ETH -> 80w 稳定币
当ETH下降 -> 清算

市场因素的分析，舆情，新闻分析虚拟货币的走势


1. LLM -> bert make sense
extraction | prediction
bert -> 轻量，逻辑上why?
2. extraction    

    key and value  analyse the meaning behind the sentence
    1. unsupervised , given label
    2. FinBert
3. GPT & Bert difference
4. LLM网络结构搭网络 
5. 训练，语料库，区块链市场中的词

![Alt text](image.png)

The structure of GPT-3 (Generative Pre-trained Transformer 3) is detailed in the paper titled "Language Models are Few-Shot Learners," published in 2020. GPT-3, developed by OpenAI, is a state-of-the-art language processing AI model known for its ability to generate human-like text based on the input it receives. It's the third iteration of the GPT series, showcasing significant improvements over its predecessors in terms of size and capability.
Key Features of GPT-3:

1. Model Size: GPT-3 is one of the largest language models ever created, with 175 billion parameters. This is a significant increase from GPT-2, which had 1.5 billion parameters.

2. Architecture: GPT-3 follows the Transformer architecture, which is based on self-attention mechanisms. This architecture was first introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017. The Transformer model has since become the foundation for many state-of-the-art NLP models, including GPT-3.

3. Training: GPT-3 is trained on a diverse range of internet text. Its training dataset includes a wide variety of genres, allowing it to have broad knowledge and the ability to generate text across different topics and styles.

4. Capabilities: GPT-3 is capable of performing a wide range of natural language processing tasks without task-specific training. This includes translation, question-answering, and text generation, among others. It can perform these tasks in a few-shot, one-shot, or zero-shot manner, meaning it can understand the task from just a few examples, a single example, or even no examples at all.

5. Applications: Due to its versatility and power, GPT-3 has been used in a variety of applications, from generating creative writing to automating customer service responses, and even coding assistance.

Graph Theory Perspective:

While GPT-3 itself is not directly based on graph theory, its underlying Transformer architecture can be understood in terms of graph theory concepts. In the Transformer, the self-attention mechanism can be thought of as creating a fully connected graph where each word (node) is connected to every other word (node) in the sentence. The attention scores represent the weights of the edges in this graph, indicating the strength of the relationship between words.

This graph-based perspective helps in understanding how the model captures the context and relationships between words in a sentence, enabling it to generate coherent and contextually relevant text.
Code Snippet:
Here's a simple Python example using the transformers library to generate text with GPT-3:

```python
from transformers import GPT3LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt3"  # This is a placeholder. Actual GPT-3 model names should be used from the Hugging Face Model Hub
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT3LMHeadModel.from_pretrained(model_name)

# Encode input context
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
```

Note: As of my last update, direct access to GPT-3 might require using OpenAI's API, and the model name in the transformers library might be different. Please check the latest documentation for accurate details.


GPT2 is open source

GPT2 -> project

I can find the top 1000 websites which consist of datasets 