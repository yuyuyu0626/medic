import torch
import pandas as pd
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 1. 配置路径 (指向你下载的 PMC-VQA 数据)
MODEL_ID = "vikhyat/moondream2"  # 极轻量模型，约 1.6B 参数
IMAGE_DIR = "/home/zhanght2504/zhanght2504_didi2/runspace_yyxs/PMC_VQA/images_2"
CSV_PATH = "/home/zhanght2504/zhanght2504_didi2/runspace_yyxs/PMC_VQA/test_2.csv"

# 2. 加载模型和分词器
print("正在加载微型模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 3. 读取数据集
df = pd.read_csv(CSV_PATH)

def predict_choice(img_name, question, choices):
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = Image.open(img_path)
    
    # 构造针对多选题的指令
    full_prompt = f"{question}\nChoices: {choices}\nAnswer with the correct option letter (A, B, C, or D) directly."
    
    # 模型推理
    enc_image = model.encode_image(image)
    answer = model.answer_question(enc_image, full_prompt, tokenizer)
    return answer

# 4. 跑前 5 条数据看看效果
print("\n--- 启动微型模型进行图像识别选择题 ---")
for i in range(5):
    row = df.iloc[i]
    # 拼接选项文本
    choices_text = f"A: {row['Choice A']}, B: {row['Choice B']}, C: {row['Choice C']}, D: {row['Choice D']}"
    
    pred = predict_choice(row['Figure_path'], row['Question'], choices_text)
    
    print(f"图片: {row['Figure_path']}")
    print(f"问题: {row['Question']}")
    print(f"模型预测: {pred}")
    print(f"标准答案: {row['Answer_label']}") # 对应 A/B/C/D
    print("-" * 30)