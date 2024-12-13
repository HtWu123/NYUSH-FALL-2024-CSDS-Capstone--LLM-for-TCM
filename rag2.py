import os
import sys
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

# Set Hugging Face cache directory
cache_dir = '/scratch/hw2933/huggingface_cache'
os.environ['HF_HOME'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

# Print cache directory confirmation
print(f"Hugging Face cache directory set to: {cache_dir}")

# File paths
retrieval_file_path = '/scratch/hw2933/new/dataset/converted_shen_nong_tcm_dataset.csv'

# Load data
retrieval_data = pd.read_csv(retrieval_file_path)

# Split data into 90% training and 10% test set
train_data = retrieval_data.sample(frac=0.9, random_state=42).reset_index(drop=True)
test_data = retrieval_data.drop(train_data.index).reset_index(drop=True)

# Prepare training and test questions and answers
train_questions = train_data['query'].tolist()
train_answers = train_data['response'].tolist()

test_questions = test_data['query'].tolist()
test_answers = test_data['response'].tolist()

# Initialize SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', cache_folder=cache_dir)
train_embeddings = model.encode(train_questions, convert_to_tensor=False)

# Initialize FAISS index
d = train_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(train_embeddings))

# Define retrieval function
def search(query, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [(train_questions[i], train_answers[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# Load generation model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modify model loading path and specify cache directory
model_dir = "/scratch/hw2933/new/model/models--hfl--llama-3-chinese-8b-instruct-v3/snapshots/e5f2d57bd555a2411c5773f64c8f2eedb95c37d0"

tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
model_llama = AutoModelForCausalLM.from_pretrained(model_dir, cache_dir=cache_dir).to(device)

# Generate answer with RAG
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

# Generate answer without RAG
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

# Compute evaluation metrics
def compute_scores(generated_answer, reference_answers):
    generated_tokens = list(jieba.cut(generated_answer))
    reference_tokens_list = [list(jieba.cut(ref)) for ref in reference_answers]
    
    bleu_score = sentence_bleu(reference_tokens_list, generated_tokens)
    
    P, R, F1 = bert_score_function([generated_answer], reference_answers, lang="zh", model_type="bert-base-chinese")
    bert_score_values = (P.mean().item(), R.mean().item(), F1.mean().item())
    
    return bleu_score, bert_score_values

# Redirect output to file
def set_output_file(file_path):
    sys.stdout = open(file_path, 'w', encoding='utf-8')

output_file_path = '/scratch/hw2933/new/results.txt'
set_output_file(output_file_path)

# Evaluation
bleu_scores_rag = []
bleu_scores_no_rag = []

bertscore_P_rag = []
bertscore_R_rag = []
bertscore_F1_rag = []

bertscore_P_no_rag = []
bertscore_R_no_rag = []
bertscore_F1_no_rag = []

# num_samples = len(test_questions) - 1
num_samples = 20
# num_samples = 5

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
    
    print(f"With RAG - BLEU: {bleu_rag}, BERTScore F1: {bert_score_rag[2]}")
    print(f"Without RAG - BLEU: {bleu_no_rag}, BERTScore F1: {bert_score_no_rag[2]}")
    print("\n")

# Calculate means and variances
def compute_mean_variance(scores):
    mean = statistics.mean(scores)
    variance = statistics.variance(scores) if len(scores) > 1 else 0
    return mean, variance

avg_bleu_rag, var_bleu_rag = compute_mean_variance(bleu_scores_rag)
avg_bertscore_P_rag, var_bertscore_P_rag = compute_mean_variance(bertscore_P_rag)
avg_bertscore_R_rag, var_bertscore_R_rag = compute_mean_variance(bertscore_R_rag)
avg_bertscore_F1_rag, var_bertscore_F1_rag = compute_mean_variance(bertscore_F1_rag)

avg_bleu_no_rag, var_bleu_no_rag = compute_mean_variance(bleu_scores_no_rag)
avg_bertscore_P_no_rag, var_bertscore_P_no_rag = compute_mean_variance(bertscore_P_no_rag)
avg_bertscore_R_no_rag, var_bertscore_R_no_rag = compute_mean_variance(bertscore_R_no_rag)
avg_bertscore_F1_no_rag, var_bertscore_F1_no_rag = compute_mean_variance(bertscore_F1_no_rag)




print("With RAG - Average and Variance:")
print(f"Average BLEU: {avg_bleu_rag}, Variance: {var_bleu_rag}")
print(f"Average BERTScore Precision: {avg_bertscore_P_rag}, Variance: {var_bertscore_P_rag}")
print(f"Average BERTScore Recall: {avg_bertscore_R_rag}, Variance: {var_bertscore_R_rag}")
print(f"Average BERTScore F1: {avg_bertscore_F1_rag}, Variance: {var_bertscore_F1_rag}")

print("\nWithout RAG - Average and Variance:")
print(f"Average BLEU: {avg_bleu_no_rag}, Variance: {var_bleu_no_rag}")
print(f"Average BERTScore Precision: {avg_bertscore_P_no_rag}, Variance: {var_bertscore_P_no_rag}")
print(f"Average BERTScore Recall: {avg_bertscore_R_no_rag}, Variance: {var_bertscore_R_no_rag}")
print(f"Average BERTScore F1: {avg_bertscore_F1_no_rag}, Variance: {var_bertscore_F1_no_rag}")

# Close output file
sys.stdout.close()
