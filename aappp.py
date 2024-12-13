import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

# 设置 Hugging Face 缓存目录
cache_dir = '/scratch/hw2933/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)
print(f"Hugging Face cache directory set to: {cache_dir}")

# 文件路径
retrieval_file_path = '/scratch/hw2933/new/dataset/converted_shen_nong_tcm_dataset.csv'

# 加载数据
retrieval_data = pd.read_csv(retrieval_file_path)

# 全部作为训练数据
train_data = retrieval_data
train_questions = train_data['query'].tolist()
train_answers = train_data['response'].tolist()

# 验证数据集完整性
assert len(train_questions) == len(train_answers), "问题和答案数量不一致！"
print(f"Loaded {len(train_questions)} questions and answers.")


# model_name = 'shibing624/text2vec-base-chinese'
# model_name = 'uer/sbert-base-chinese-nli'
model_name = 'shibing624/text2vec-base-chinese'

model = SentenceTransformer(model_name, cache_folder=cache_dir)
print(f"SentenceTransformer model '{model_name}' loaded.")


train_embeddings = model.encode(train_questions, convert_to_tensor=False)
train_embeddings = np.array(train_embeddings)

# 初始化 FAISS 索引
d = train_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(train_embeddings)
print(f"FAISS index initialized with {index.ntotal} vectors.")

# 定义检索函数
def search(query, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=False)
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # 确保 query_embedding 是二维的
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)
    results = []
    for j, i in enumerate(indices[0]):
        if i != -1:
            results.append((train_questions[i], train_answers[i], distances[0][j]))
    print(f"Search results: {results}")
    return results

# 加载生成模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = "/scratch/hw2933/new/model/models--hfl--llama-3-chinese-8b-instruct-v3/snapshots/e5f2d57bd555a2411c5773f64c8f2eedb95c37d0"

tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
model_llama = AutoModelForCausalLM.from_pretrained(model_dir, cache_dir=cache_dir).to(device)
print("Generation model loaded successfully.")

# 生成包含 RAG 的答案
def generate_answer(query, top_k=5):
    results_R = search(query, top_k=top_k)
    print(f"Retrieved {len(results_R)} results.")
    
    # 构建相关上下文
    relevant_context = "\n".join([f"相关信息 {i+1}：{result[1]}" for i, result in enumerate(results_R)])
    context = (
        f"请根据以下信息回答问题：\n"
        f"问题：{query}\n"
        f"{relevant_context}\n"
        f"回答："
    )
    
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    generated_ids = model_llama.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    answer = generated_text.split("回答：")[-1].strip()
    print(f"Generated answer (with RAG): {answer}")
    return answer, results_R

# 生成不包含 RAG 的答案
def generate_answer_no_rag(query):
    context = f"请回答以下问题：\n问题：{query}\n回答："
    input_ids = tokenizer.encode(context, return_tensors="pt").to(device)
    generated_ids = model_llama.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    answer = generated_text.split("回答：")[-1].strip()
    print(f"Generated answer (no RAG): {answer}")
    return answer

# 定义 Gradio 接口函数
def rag_interface(query, use_rag):
    if not query.strip():
        return "请输入一个有效的问题！Please enter a valid question!", "无相关信息 No relevant information"
    
    if use_rag:
        answer, results_R = generate_answer(query)
        retrieved_info = "\n".join([
            f"【相关问题 {idx+1}】{q}\n【相关答案】{a}\n【距离】{dist:.4f}"
            for idx, (q, a, dist) in enumerate(results_R)
        ])
        return answer, retrieved_info
    else:
        answer = generate_answer_no_rag(query)
        return answer, "无相关信息 No relevant information"

# 创建 Gradio 接口
interface = gr.Interface(
    fn=rag_interface,
    inputs=[
        gr.Textbox(lines=2, placeholder='请输入你的问题 Please input your question', label='问题 Question'),
        gr.Checkbox(value=True, label='使用 RAG（检索增强生成） Using RAG (Retrieval-Augmented Generation)')
    ],
    outputs=[
        gr.Textbox(label='生成的答案 Generated answer'),
        gr.Textbox(label='检索到的相关信息 Retrieved relevant information ', lines=10)
    ],
    title='Traditional Chinese Medicine LLM 中医大模型',
    description='你有什么不舒服的地方，输入你的问题。Do you have any discomfort, enter your question.'
)

# 启动 Gradio 接口
interface.launch(share=True)
