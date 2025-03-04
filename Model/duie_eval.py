import re
import ast
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "/data/home/Guanchao/models/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# file_path = "../KG Construction/DuIE2.0/prompts/duie-0-shot-prompt.txt"
# file_path = "../KG Construction/DuIE2.0/prompts/duie-0-shot-prompt-opt.txt"
# file_path = "../KG Construction/DuIE2.0/prompts/duie-1-shot-prompt.txt"
file_path = "../KG Construction/DuIE2.0/prompts/duie-1-shot-prompt-opt.txt"
with open(file_path, "r") as f:
    content = f.read()

prompts = re.split(r"\n\d+\n", content)

# 初始化问题集合和答案集合
questions = []
answers = []

def fix_spo(spo_str):
    """
    将类似 [抄底逃顶实战技巧 , 作者 , 郝鸿雁 ] 的字符串转换为合法的 Python 字面量：
    ['抄底逃顶实战技巧', '作者', '郝鸿雁']
    """
    # 去除前后空格和中括号
    inner = spo_str.strip()[1:-1]
    # 按逗号分割各个项
    tokens = [token.strip() for token in inner.split(',')]
    # 对每个token加上引号
    tokens = [f"'{token}'" for token in tokens if token]
    fixed_str = "[" + ", ".join(tokens) + "]"
    return fixed_str

def remove_symbols(text):
    # 使用正则表达式去除所有《》“”‘’"'"符号
    return re.sub(r"[《》“”‘’'\"']", "", text)


for prompt in prompts:
    # question_match = re.search(r"(已知候选谓词列表：.*?SPO三元组:)", prompt, re.DOTALL)
    question_match = re.search(r"已知候选谓词列表：(.+?)Duie2.0 :", prompt, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()  # 去除前后空格
        questions.append(question)

    answer_match = re.search(r"Duie2.0\s*:\s*(.*)", prompt, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()

        # 提取 `[主语, 谓语, 宾语]` 格式的 SPO 三元组
        answer_spo_list = re.findall(r"\[.*?\]", answer_text)
        answer_spo_list = [ast.literal_eval(fix_spo(spo)) for spo in answer_spo_list]
        answers.append(answer_spo_list)
    else:
        answers.append([])

# 输出提取的前5个示例
print("=== 提取的问题集合（前5个示例） ===")
for idx, q in enumerate(questions[:5]):
    print(f"\n问题 {idx + 1}:\n{q}")

print("\n=== 提取的答案集合（前5个示例） ===")
for idx, a in enumerate(answers[:5]):
    print(f"\n答案 {idx + 1}:\n{a}")

def extract_spo(output):
    """
    提取文本中的三元组并返回每个三元组作为列表的形式
    """
    inner = output.strip()[1:-1]
    # 找到所有的三元组字符串形式
    output_spo_list = re.findall(r"\[.*?\]", inner)

    # 修复格式并转化为列表
    output_spo_list = [ast.literal_eval(fix_spo(spo)) for spo in output_spo_list]

    return output_spo_list

outputs = []

for i, question in enumerate(questions):
    messages = [
        {"role": "system", "content": "你是一个帮助进行构建知识图谱的机器人，主要任务是进行实体关系的提取与构建"},
        {"role": "user", "content": question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = remove_symbols(response)
    print(response)
    output_spo_list = extract_spo(response)
    print(output_spo_list)
    outputs.append(output_spo_list)


def calculate_f1(y_true, y_pred):
    """
    计算 Precision, Recall, F1 Score
    :param y_true: 标准答案三元组列表
    :param y_pred: 模型预测三元组列表
    :return: (precision, recall, f1)
    """
    y_true_set = set(map(tuple, y_true))  # 转换为集合以加速查找
    y_pred_set = set(map(tuple, y_pred))

    tp = len(y_true_set & y_pred_set)  # 计算正确匹配的三元组数
    pred_count = len(y_pred_set) or 1  # 避免分母为 0
    true_count = len(y_true_set) or 1

    precision = tp / pred_count
    recall = tp / true_count
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# 计算每组数据的 Precision、Recall、F1 Score
metrics = [calculate_f1(true, pred) for true, pred in zip(answers, outputs)]
precision_list, recall_list, f1_list = zip(*metrics)  # 解包列表

# 计算整体的平均值
average_precision = np.mean(precision_list)
average_recall = np.mean(recall_list)
average_f1 = np.mean(f1_list)

print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average F1 Score: {average_f1:.4f}")
