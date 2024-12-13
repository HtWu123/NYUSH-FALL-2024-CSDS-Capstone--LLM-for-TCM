from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "hfl/llama-3-chinese-8b-instruct-v3"
model_dir = "/scratch/hw2933/new/model"
# 下载模型
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_dir)  # 下载模型
# 保存模型
print("模型下载并保存到指定目录。")
