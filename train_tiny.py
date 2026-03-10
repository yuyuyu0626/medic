import os
import sys
import torch
import pandas as pd
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm  # 如果报错请 pip install tqdm

# 1. 路径配置
MODEL_PATH = "/home/zhanght2504/zhanght2504_didi2/runspace_yyxs/test/pretrain_model"
IMAGE_DIR = "/home/zhanght2504/zhanght2504_didi2/runspace_yyxs/PMC_VQA/images_2/figures"
CSV_PATH = "/home/zhanght2504/zhanght2504_didi2/runspace_yyxs/PMC_VQA/test_2.csv"
OUTPUT_CSV = "./pmc_vqa_results.csv" # 结果保存路径

# 2. 环境准备
sys.path.append(MODEL_PATH)
os.chdir(MODEL_PATH)

from moondream import MoondreamModel
from config import MoondreamConfig
from transformers import AutoTokenizer

# 3. 初始化模型
print("正在初始化本地模型...")
config = MoondreamConfig() 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MoondreamModel(config).to(device)

print("载入权重中...")
weights_path = os.path.join(MODEL_PATH, "model.safetensors")
model.load_state_dict(load_file(weights_path), strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)

# 4. 加载数据
df = pd.read_csv(CSV_PATH).sample(1)
# 如果你想先跑一小部分测试，可以取消下面这行的注释：
# df = df.head(500) 

# 5. 定义单条推理逻辑
def predict_vqa(row):
    img_path = os.path.join(IMAGE_DIR, row['Figure_path'])
    if not os.path.exists(img_path):
        return "ERROR_IMG_NOT_FOUND"
    
    try:
        image = Image.open(img_path).convert("RGB")
        question = row['Question']
        choices = f"A: {row['Choice A']}, B: {row['Choice B']}, C: {row['Choice C']}, D: {row['Choice D']}"
        
        # 强制模型只输出单个字母
        prompt = f"Question: {question}\nChoices: {choices}\nOutput the correct option letter (A, B, C, or D) only."
        
        with torch.no_grad():
            result = model.query(image=image, question=prompt)
            # 简单清洗结果，只保留第一个字母并大写
            ans = result["answer"].strip().upper()
            return ans[0] if len(ans) > 0 else "N/A"
    except Exception as e:
        return f"ERROR_{str(e)}"

# 6. 开始批量运行
print(f"开始批量推理，共 {len(df)} 条数据...")
results = []
correct_count = 0

# 使用 tqdm 显示进度条
pbar = tqdm(df.iterrows(), total=len(df))
for index, row in pbar:
    prediction = predict_vqa(row)
    actual = str(row['Answer']).strip().upper()
    
    # 计算实时准确率
    is_correct = 1 if prediction == actual else 0
    correct_count += is_correct
    current_acc = correct_count / (len(results) + 1)
    
    # 记录结果
    results.append({
        "Figure_path": row['Figure_path'],
        "Question": row['Question'],
        "Actual": actual,
        "Predicted": prediction,
        "Is_Correct": is_correct
    })
    
    # 更新进度条显示的准确率
    pbar.set_description(f"Acc: {current_acc:.2%}")
    
    # 每 100 条保存一次，防止白干
    if len(results) % 100 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

# 7. 最终保存
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n任务完成！最终准确率: {correct_count/len(df):.2%}")
print(f"结果已保存至: {OUTPUT_CSV}")