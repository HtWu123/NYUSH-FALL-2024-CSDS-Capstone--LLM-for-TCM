import os
import sys
import seaborn as sns
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nltk.translate.bleu_score import sentence_bleu
from bert_score import score as bert_score_function
import jieba
import statistics
import matplotlib.pyplot as plt

# 设置 Hugging Face 缓存目录
cache_dir = '/scratch/hw2933/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# 打印缓存目录确认
print(f"Hugging Face cache directory set to: {cache_dir}")

# 文件路径
retrieval_file_path = '/scratch/hw2933/new/dataset/converted_shen_nong_tcm_dataset.csv'
test_file_path = '/scratch/hw2933/new/dataset/huatuo_test.csv'
# 加载数据
retrieval_data = pd.read_csv(retrieval_file_path)
train_data = retrieval_data
test_data = pd.read_csv(test_file_path, encoding='utf-8')
# 将数据划分为 90% 的训练集和 10% 的测试集

# 准备训练和测试问题及答案
train_questions = train_data['query'].tolist()
train_answers = train_data['response'].tolist()

test_questions = test_data['questions'].tolist()
test_answers = test_data['answers'].tolist()

# 初始化适用于中文的 SentenceTransformer 模型
model_name = 'shibing624/text2vec-base-chinese'
model = SentenceTransformer(model_name, cache_folder=cache_dir)
print(f"SentenceTransformer model '{model_name}' loaded.")

# 对训练问题进行编码
train_embeddings = model.encode(train_questions, convert_to_numpy=True)

# 初始化 FAISS 索引
d = train_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(train_embeddings)
print(f"FAISS index initialized with {index.ntotal} vectors.")

# 定义检索函数
def search(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
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

# 修改模型加载路径并指定缓存目录
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
    return answer  # 只返回答案字符串

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

# 计算评估指标
def compute_scores(generated_answer, reference_answers):
    generated_tokens = list(jieba.cut(generated_answer))
    reference_tokens_list = [list(jieba.cut(ref)) for ref in reference_answers]
    
    bleu_score = sentence_bleu(reference_tokens_list, generated_tokens)
    
    P, R, F1 = bert_score_function([generated_answer], reference_answers, lang="zh", model_type="bert-base-chinese")
    bert_score_values = (P.mean().item(), R.mean().item(), F1.mean().item())
    
    return bleu_score, bert_score_values

# 重定向输出到文件
def set_output_file(file_path):
    sys.stdout = open(file_path, 'w', encoding='utf-8')

output_file_path = '/scratch/hw2933/new/nrag_results.txt'
set_output_file(output_file_path)

# 评估
bleu_scores_rag, bleu_scores_no_rag = [], []
bertscore_P_rag, bertscore_R_rag, bertscore_F1_rag = [], [], []
bertscore_P_no_rag, bertscore_R_no_rag, bertscore_F1_no_rag = [], [], []

# 评估的样本数量
# num_samples = len(test_questions) - 1
num_samples = 10
print(f"Evaluating on {num_samples} samples.")


for i in range(num_samples):
    query = test_questions[i]
    reference_answers = [test_answers[i]]
    print(f"Processing question {i+1}: {query}")
    
    # With RAG
    generated_answer_rag = generate_answer(query, top_k=5)
    bleu_rag, bert_score_rag = compute_scores(generated_answer_rag, reference_answers)
    bleu_scores_rag.append(bleu_rag)
    bertscore_P_rag.append(bert_score_rag[0])
    bertscore_R_rag.append(bert_score_rag[1])
    bertscore_F1_rag.append(bert_score_rag[2])
    
    # Without RAG
    generated_answer_no_rag = generate_answer_no_rag(query)
    bleu_no_rag, bert_score_no_rag = compute_scores(generated_answer_no_rag, reference_answers)
    bleu_scores_no_rag.append(bleu_no_rag)
    bertscore_P_no_rag.append(bert_score_no_rag[0])
    bertscore_R_no_rag.append(bert_score_no_rag[1])
    bertscore_F1_no_rag.append(bert_score_no_rag[2])

# 计算均值和方差
def compute_mean_variance(scores):
    mean = statistics.mean(scores)
    variance = statistics.variance(scores) if len(scores) > 1 else 0
    return mean, variance

# 计算并打印平均值和方差
# With RAG
avg_bleu_rag, var_bleu_rag = compute_mean_variance(bleu_scores_rag)
avg_bertscore_P_rag, var_bertscore_P_rag = compute_mean_variance(bertscore_P_rag)
avg_bertscore_R_rag, var_bertscore_R_rag = compute_mean_variance(bertscore_R_rag)
avg_bertscore_F1_rag, var_bertscore_F1_rag = compute_mean_variance(bertscore_F1_rag)

# Without RAG
avg_bleu_no_rag, var_bleu_no_rag = compute_mean_variance(bleu_scores_no_rag)
avg_bertscore_P_no_rag, var_bertscore_P_no_rag = compute_mean_variance(bertscore_P_no_rag)
avg_bertscore_R_no_rag, var_bertscore_R_no_rag = compute_mean_variance(bertscore_R_no_rag)
avg_bertscore_F1_no_rag, var_bertscore_F1_no_rag = compute_mean_variance(bertscore_F1_no_rag)

# 打印结果到文件
print("\nWith RAG - Average and Variance:")
print(f"Average BLEU: {avg_bleu_rag:.4f}, Variance: {var_bleu_rag:.4f}")
print(f"Average BERTScore Precision: {avg_bertscore_P_rag:.4f}, Variance: {var_bertscore_P_rag:.4f}")
print(f"Average BERTScore Recall: {avg_bertscore_R_rag:.4f}, Variance: {var_bertscore_R_rag:.4f}")
print(f"Average BERTScore F1: {avg_bertscore_F1_rag:.4f}, Variance: {var_bertscore_F1_rag:.4f}")

print("\nWithout RAG - Average and Variance:")
print(f"Average BLEU: {avg_bleu_no_rag:.4f}, Variance: {var_bleu_no_rag:.4f}")
print(f"Average BERTScore Precision: {avg_bertscore_P_no_rag:.4f}, Variance: {var_bertscore_P_no_rag:.4f}")
print(f"Average BERTScore Recall: {avg_bertscore_R_no_rag:.4f}, Variance: {var_bertscore_R_no_rag:.4f}")
print(f"Average BERTScore F1: {avg_bertscore_F1_no_rag:.4f}, Variance: {var_bertscore_F1_no_rag:.4f}")

# 绘制结果函数
# def plot_results(metric_name, rag_scores, no_rag_scores, save_path):
#     rag_mean, rag_var = compute_mean_variance(rag_scores)
#     no_rag_mean, no_rag_var = compute_mean_variance(no_rag_scores)

#     labels = ['With RAG', 'Without RAG']
#     means = [rag_mean, no_rag_mean]
#     variances = [rag_var, no_rag_var]

#     plt.figure(figsize=(8, 6))
#     plt.bar(labels, means, yerr=np.sqrt(variances), capsize=10)
#     plt.ylabel(metric_name)
#     plt.title(f'{metric_name} Comparison')
#     plt.savefig(save_path)
#     plt.close()

def plot_results(metric_name, rag_scores, no_rag_scores, save_path):
    # Compute means and standard deviations
    rag_mean, rag_var = compute_mean_variance(rag_scores)
    no_rag_mean, no_rag_var = compute_mean_variance(no_rag_scores)
    rag_std = np.sqrt(rag_var)
    no_rag_std = np.sqrt(no_rag_var)

    # Prepare data for plotting
    data = pd.DataFrame({
        'Method': ['With RAG', 'Without RAG'],
        'Mean Score': [rag_mean, no_rag_mean],
        'Std Dev': [rag_std, no_rag_std]
    })

    # Set plot style
    sns.set(style='whitegrid')

    # Create the barplot
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='Method', y='Mean Score', data=data, errorbar=None, palette='viridis')  # Updated ci and palette

    # Add error bars
    ax.errorbar(x=[0, 1], y=data['Mean Score'], yerr=data['Std Dev'], fmt='none', c='black', capsize=10)

    # Set labels and title
    plt.ylabel(metric_name, fontsize=14)
    plt.xlabel('Method', fontsize=14)
    plt.title(f'{metric_name} Comparison', fontsize=16)

    # Adjust tick parameters
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



# # 生成图表
# plot_results("BLEU Score", bleu_scores_rag, bleu_scores_no_rag, "/scratch/hw2933/new/bleu_score_comparison.png")
# plot_results("BERTScore Precision", bertscore_P_rag, bertscore_P_no_rag, "/scratch/hw2933/new/bertscore_precision_comparison.png")
# plot_results("BERTScore Recall", bertscore_R_rag, bertscore_R_no_rag, "/scratch/hw2933/new/bertscore_recall_comparison.png")
# plot_results("BERTScore F1", bertscore_F1_rag, bertscore_F1_no_rag, "/scratch/hw2933/new/bertscore_f1_comparison.png")

plot_results("BLEU Score", bleu_scores_rag, bleu_scores_no_rag, "/scratch/hw2933/new/1bleu_score_comparison.png")
plot_results("BERTScore Precision", bertscore_P_rag, bertscore_P_no_rag, "/scratch/hw2933/new/1bertscore_precision_comparison.png")
plot_results("BERTScore Recall", bertscore_R_rag, bertscore_R_no_rag, "/scratch/hw2933/new/1bertscore_recall_comparison.png")
plot_results("BERTScore F1", bertscore_F1_rag, bertscore_F1_no_rag, "/scratch/hw2933/new/1bertscore_f1_comparison.png")

print("Plots saved successfully.")

# 关闭输出文件
sys.stdout.close()
