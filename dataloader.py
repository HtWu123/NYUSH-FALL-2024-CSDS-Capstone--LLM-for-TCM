# from datasets import Dataset
# import json

# # 1. 加载数据集
# dataset = Dataset.from_file("/scratch/hw2933/capstone/dataset/shen_nong_tcm_dataset-train.arrow")

# qa_pairs = []

# # 遍历数据集，提取问题和答案对
# for i in range(len(dataset)):
#     question = dataset[i]['query']  # 假设 'question' 是问题的字段名
#     answer = dataset[i]['response']      # 假设 'answer' 是答案的字段名
#     qa_pairs.append({'question': question, 'answer': answer})



# with open("/scratch/hw2933/capstone/dataset/tcm_qa.json", "w", encoding='utf-8') as f:
#     json.dump({'train': qa_pairs}, f, ensure_ascii=False, indent=4)


# import json
# import random

# # Load JSON data from file
# file_path = "/scratch/hw2933/new/dataset/tcm_qa.json"

# with open(file_path, 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Print the data structure to understand it
# print(f"Data type: {type(data)}")  # Confirm that data is a dict
# print(f"Keys: {data.keys()}")  # Check the keys in the dict

# # Extract the 'train' key
# train_data = data.get('train', [])

# # Print the length of the train data to see if it's populated
# print(f"Train data length: {len(train_data)}")

# # Proceed only if train_data is not empty
# if isinstance(train_data, list) and len(train_data) > 0:
#     # Shuffle and split the dataset
#     random.shuffle(train_data)
#     split_idx = int(len(train_data) * 0.8)
#     train_split = train_data[:split_idx]
#     valid_split = train_data[split_idx:]

#     print(f"Train split entries: {len(train_split)}, Validation split entries: {len(valid_split)}")

#     # Save the train and validation datasets back to JSON files
#     train_file_path = "/scratch/hw2933/new/dataset/train.json"
#     valid_file_path = "/scratch/hw2933/new/dataset/valid.json"

#     with open(train_file_path, 'w', encoding='utf-8') as train_file:
#         json.dump(train_split, train_file, ensure_ascii=False, indent=4)

#     with open(valid_file_path, 'w', encoding='utf-8') as valid_file:
#         json.dump(valid_split, valid_file, ensure_ascii=False, indent=4)

# else:
#     print("Error: 'train' data is empty or not a list.")


from datasets import Dataset
import pandas as pd

# 加载 Hugging Face .arrow 文件
arrow_file_path = "/scratch/hw2933/new/dataset/shen_nong_tcm_dataset-train.arrow"
dataset = Dataset.from_file(arrow_file_path)

# 将 Dataset 转换为 Pandas DataFrame
df = pd.DataFrame(dataset)

# 保存为 CSV 文件
csv_output_path = "/scratch/hw2933/new/dataset/converted_shen_nong_tcm_dataset.csv"
df.to_csv(csv_output_path, index=False)

print(f"CSV 文件保存为: {csv_output_path}")

