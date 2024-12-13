import os
import sys  # 新增
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nltk.translate.bleu_score import sentence_bleu
# from rouge import Rouge  # 移除 ROUGE 的导入
from bert_score import score as bert_score_function
import jieba
import statistics  # 新增，用于计算方差

# 设置 Hugging Face 缓存路径
cache_dir = '/scratch/hw2933/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# 打印确认缓存路径
print(f"Hugging Face 缓存路径设置为: {cache_dir}")

# 文件路径
retrieval_file_path = '/scratch/hw2933/new/dataset/converted_shen_nong_tcm_dataset.csv'
test_file_path = '/scratch/hw2933/new/dataset/huatuo_test.csv'

# 读取数据
retrieval_data = pd.read_csv(retrieval_file_path)
test_data = pd.read_csv(test_file_path)

retrieval_questions = retrieval_data['query'].tolist()
retrieval_answers = retrieval_data['response'].tolist()

test_questions = test_data['questions'].tolist()
test_answers = test_data['answers'].tolist()

# 初始化 SentenceTransformer 模型
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder=cache_dir)
retrieval_embeddings = model.encode(retrieval_questions, convert_to_tensor=False)

# 初始化 FAISS 索引
d = retrieval_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(retrieval_embeddings))

# 定义检索函数
def search(query, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [(retrieval_questions[i], retrieval_answers[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# 加载生成模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 修改分词器和模型的加载路径，同时指定 cache_dir
model_dir = "/scratch/hw2933/new/model/models--hfl--llama-3-chinese-8b-instruct-v3/snapshots/e5f2d57bd555a2411c5773f64c8f2eedb95c37d0"

tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
model_llama = AutoModelForCausalLM.from_pretrained(model_dir, cache_dir=cache_dir).to(device)

# 生成答案（带 RAG）
def generate_answer(query, top_k=3):
    results = search(query, top_k=top_k)
    print('Search 完成')
    print(f"共检索到 {top_k} 个结果。")
    
    # 使用最相关的检索结果
    most_relevant = results[0]
    
    # 构建新的 prompt，添加明确的回答标记
    context = (
        f"请根据以下信息回答问题：\n"
        f"问题：{query}\n"
        f"相关信息：{most_relevant[1]}\n"
        f"回答："
    )
    
    # 可选择是否打印生成的 prompt
    # print(f"生成的 prompt: {context}")
    
    input_text = context
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # 设置生成参数，添加 eos_token_id
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
    
    # 提取“回答：”之后的内容，作为生成的答案
    answer = generated_text.split("回答：")[-1].strip()
    
    # 打印生成的答案
    print(f"生成的答案（带 RAG）：{answer}")
    
    return answer

# 生成答案（不带 RAG）
def generate_answer_no_rag(query):
    # 构建 prompt，不包含检索信息
    context = (
        f"请回答以下问题：\n"
        f"问题：{query}\n"
        f"回答："
    )
    
    # 可选择是否打印生成的 prompt
    # print(f"生成的 prompt（无 RAG）: {context}")
    
    input_text = context
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # 设置生成参数，添加 eos_token_id
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
    
    # 提取“回答：”之后的内容，作为生成的答案
    answer = generated_text.split("回答：")[-1].strip()
    
    # 打印生成的答案
    print(f"生成的答案（无 RAG）：{answer}")
    
    return answer

# 计算评价指标
def compute_scores(generated_answer, reference_answers):
    # 使用 Jieba 对中文文本进行分词
    generated_tokens = list(jieba.cut(generated_answer))
    reference_tokens_list = [list(jieba.cut(ref)) for ref in reference_answers]
    
    # 计算 BLEU 分数
    bleu_score = sentence_bleu(reference_tokens_list, generated_tokens)
    
    # 计算 BERTScore
    P, R, F1 = bert_score_function([generated_answer], reference_answers, lang="zh", model_type="bert-base-chinese")
    bert_score_values = (P.mean().item(), R.mean().item(), F1.mean().item())
    
    return bleu_score, bert_score_values

# 定义函数来设置输出文件
def set_output_file(file_path):
    sys.stdout = open(file_path, 'w', encoding='utf-8')

# 设置输出文件路径
output_file_path = '/scratch/hw2933/new/results.txt'  # 请将此路径替换为你想要保存输出的文件路径

# 重定向标准输出到文件
set_output_file(output_file_path)

# 执行评估
# 初始化累积变量
bleu_scores_rag = []
bleu_scores_no_rag = []

bertscore_P_rag = []
bertscore_R_rag = []
bertscore_F1_rag = []

bertscore_P_no_rag = []
bertscore_R_no_rag = []
bertscore_F1_no_rag = []

num_samples = len(test_questions) - 1  # 测试集中的问题数量
# num_samples = 10  # 你可以根据需要修改样本数量

for i in range(num_samples):
    query = test_questions[i]
    reference_answers = [test_answers[i]]  # 假设每个问题只有一个参考答案
    
    print(f"处理第 {i+1} 个问题: {query}")
    
    # 生成答案（带 RAG）
    generated_answer_rag = generate_answer(query, top_k=3)
    # 计算评价指标
    bleu_rag, bert_score_rag = compute_scores(generated_answer_rag, reference_answers)
    
    # 累加分数
    bleu_scores_rag.append(bleu_rag)
    bertscore_P_rag.append(bert_score_rag[0])
    bertscore_R_rag.append(bert_score_rag[1])
    bertscore_F1_rag.append(bert_score_rag[2])
    
    # 生成答案（不带 RAG）
    generated_answer_no_rag = generate_answer_no_rag(query)
    # 计算评价指标
    bleu_no_rag, bert_score_no_rag = compute_scores(generated_answer_no_rag, reference_answers)
    
    # 累加分数
    bleu_scores_no_rag.append(bleu_no_rag)
    bertscore_P_no_rag.append(bert_score_no_rag[0])
    bertscore_R_no_rag.append(bert_score_no_rag[1])
    bertscore_F1_no_rag.append(bert_score_no_rag[2])
    
    # 可选：打印中间结果
    print(f"带 RAG - BLEU: {bleu_rag}, BERTScore F1: {bert_score_rag[2]}")
    print(f"不带 RAG - BLEU: {bleu_no_rag}, BERTScore F1: {bert_score_no_rag[2]}")
    print("\n")

# 计算平均分数和方差
def compute_mean_variance(scores):
    mean = statistics.mean(scores)
    variance = statistics.variance(scores) if len(scores) > 1 else 0
    return mean, variance

# 计算带 RAG 的平均值和方差
avg_bleu_rag, var_bleu_rag = compute_mean_variance(bleu_scores_rag)
avg_bertscore_P_rag, var_bertscore_P_rag = compute_mean_variance(bertscore_P_rag)
avg_bertscore_R_rag, var_bertscore_R_rag = compute_mean_variance(bertscore_R_rag)
avg_bertscore_F1_rag, var_bertscore_F1_rag = compute_mean_variance(bertscore_F1_rag)

# 计算不带 RAG 的平均值和方差
avg_bleu_no_rag, var_bleu_no_rag = compute_mean_variance(bleu_scores_no_rag)
avg_bertscore_P_no_rag, var_bertscore_P_no_rag = compute_mean_variance(bertscore_P_no_rag)
avg_bertscore_R_no_rag, var_bertscore_R_no_rag = compute_mean_variance(bertscore_R_no_rag)
avg_bertscore_F1_no_rag, var_bertscore_F1_no_rag = compute_mean_variance(bertscore_F1_no_rag)

# 打印结果
print("带 RAG 的平均值和方差：")
print(f"平均 BLEU: {avg_bleu_rag}, 方差: {var_bleu_rag}")
print(f"平均 BERTScore Precision: {avg_bertscore_P_rag}, 方差: {var_bertscore_P_rag}")
print(f"平均 BERTScore Recall: {avg_bertscore_R_rag}, 方差: {var_bertscore_R_rag}")
print(f"平均 BERTScore F1: {avg_bertscore_F1_rag}, 方差: {var_bertscore_F1_rag}")

print("\n不带 RAG 的平均值和方差：")
print(f"平均 BLEU: {avg_bleu_no_rag}, 方差: {var_bleu_no_rag}")
print(f"平均 BERTScore Precision: {avg_bertscore_P_no_rag}, 方差: {var_bertscore_P_no_rag}")
print(f"平均 BERTScore Recall: {avg_bertscore_R_no_rag}, 方差: {var_bertscore_R_no_rag}")
print(f"平均 BERTScore F1: {avg_bertscore_F1_no_rag}, 方差: {var_bertscore_F1_no_rag}")

# 在程序结束时，关闭标准输出文件
sys.stdout.close()