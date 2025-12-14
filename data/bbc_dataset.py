import pandas as pd  # csv处理
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re

categories = ["business", "entertainment", "politics", "sport", "tech"]

minimindtokenizer = AutoTokenizer.from_pretrained("../model/")


# 将category的name转化为标签
def get_category_id(category_name: str) -> int:
    return categories.index(category_name)


# 清洗
def clean_text(text: str) -> str:
    if pd.isna(text):  # 对缺失值返回空字符串
        return ""

    text = str(text).lower()

    # 正则匹配去除url
    text = re.sub(r"http\S+|www.\S+", "", text, flags=re.MULTILINE)

    # 正则匹配去除空白字符换为空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


class BBCNewsDataset(Dataset):
    def __init__(self, csv_file_path: str, max_length: int = 512):
        print(f"Loading data from: {csv_file_path}")

        # 设置token列的最大长度
        self.max_length = max_length

        # 读取csv存入data属性，并按照标准重命名
        self.data = pd.read_csv(
            csv_file_path, usecols=["category", "content"], sep="\t"
        )
        self.data.rename(columns={"content": "text"}, inplace=True)

        # 数据清洗和检查
        self.data.dropna(subset=["text", "category"], inplace=True)
        print(f"Dataset initialized with {len(self.data)} valid samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        category_name = row["category"]

        # 数据清洗
        text_cleaned = clean_text(text)

        # 将name转换为数字标签
        label_id = get_category_id(category_name)

        # 使用tokenizer将数据转化为长度为max_length的token id列和attention掩码列两个tensor
        tokenized_data = minimindtokenizer(
            text_cleaned,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized_data["input_ids"].squeeze(0)
        attention_mask = tokenized_data["attention_mask"].squeeze(0)

        return (
            input_ids,
            torch.tensor(label_id, dtype=torch.long),
        )


if __name__ == "__main__":
    pass
