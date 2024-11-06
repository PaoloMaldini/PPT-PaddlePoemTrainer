import pickle
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import ErnieTokenizer, ErnieModel
import paddle.nn as nn
import numpy as np
import gradio as gr
from datetime import datetime
import os
import requests
import json
import time
class ErnieForSequenceRegression(nn.Layer):
    def __init__(self, ernie):
        super().__init__()
        self.ernie = ernie
        self.regressor = nn.Linear(self.ernie.config["hidden_size"], 1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        sequence_output, pooled_output = self.ernie(input_ids, token_type_ids=token_type_ids,
                                                    position_ids=position_ids, attention_mask=attention_mask)
        regression_output = self.regressor(pooled_output)
        similarity_score = nn.functional.sigmoid(regression_output)
        return similarity_score
import re

# 初始化 Tokenizer 和模型
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
ernie = ErnieModel.from_pretrained('ernie-1.0')
model = ErnieForSequenceRegression(ernie)

# 加载模型函数
def load_model(model, model_path):
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    return model

# 加载训练好的最佳模型
best_model_path = "best_model_epoch_1.pdparams" 
model = load_model(model, best_model_path)

def precompute_embeddings(model, tokenizer, vocab):
    model.eval()
    embeddings = {}
    batch_size = 32
    
    with paddle.no_grad():
        for i in range(0, len(vocab), batch_size):
            batch = vocab[i:i+batch_size]
            encoded_inputs = tokenizer(text=batch, max_length=128, padding=True, truncation=True, return_tensors="pd")
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            
            _, pooled_output = model.ernie(input_ids, token_type_ids=token_type_ids)
            
            for j, word in enumerate(batch):
                embeddings[word] = pooled_output[j].numpy()
    
    return embeddings

# 加载词汇表
file_path = 'words.txt'
def load_vocab(file_path):
    vocab = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip()  # 去除行首行尾的空白字符
                if word:  # 如果行不为空，则添加到词汇表中
                    vocab.append(word)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
    return vocab

vocab = load_vocab(file_path)
# 预计算词嵌入
precomputed_embeddings = precompute_embeddings(model, tokenizer, vocab)

# 查找相似词函数
def find_similar_words(input_word, top_k=5):
    model.eval()
    
    with paddle.no_grad():
        input_encoded = tokenizer(text=input_word, max_length=128, truncation=True, return_tensors="pd")
        input_ids = input_encoded["input_ids"]
        token_type_ids = input_encoded["token_type_ids"]
        
        _, input_embedding = model.ernie(input_ids, token_type_ids=token_type_ids)
        input_embedding = input_embedding.numpy()

    similarities = []
    for word, embedding in precomputed_embeddings.items():
        if word == input_word:
            continue
        
        similarity = np.dot(input_embedding, embedding.T) / (np.linalg.norm(input_embedding) * np.linalg.norm(embedding))
        similarities.append((word, similarity[0]))

    # 排序并选择 top_k 个结果
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_results = similarities[:top_k]
    
    # 格式化结果
    result = ""
    for word, sim in top_results:
        result += f"{word}\n"
    
    return result


API_KEY = "uj5JrsYncv5BMlzulVc3Jp1r"
SECRET_KEY = "pJTuRdLaYWzlFBnsLwXqmbyF9uYyAYxh"

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

def generate_image(text):
    url = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2imgv2?access_token=" + get_access_token()

    payload = json.dumps({
        "prompt": text,
        "width": 1024,
        "height": 1024
    })
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload).json()
    
    task_id = response['data']['task_id']

    url2 = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImgv2?access_token=" + get_access_token()
    
    payload2 = json.dumps({
        "task_id": task_id
    })
    time.sleep(20)
    
    response2 = requests.request("POST", url2, headers=headers, data=payload2).json()

    img_url = response2['data']['sub_task_result_list'][0]['final_image_list'][0]['img_url']
    img_response = requests.get(img_url)
    
    if img_response.status_code == 200:
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = './outputs'
        file_name = f"{current_time}.png"
        save_path = os.path.join(output_folder, file_name)

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 保存图片
        with open(save_path, "wb") as f:
            f.write(img_response.content)

    return save_path

word2vec_file = "word_vec_SC.txt"
word_to_idx_file = "SC_word_to_idx.pkl"
idx_to_word_file = "SC_idx_to_word.pkl"

# 加载词汇映射
with open(word_to_idx_file, 'rb') as f:
    word_to_idx = pickle.load(f)

with open(idx_to_word_file, 'rb') as f:
    idx_to_word = pickle.load(f)

# 加载预训练的词向量
def load_word_vectors(file_path):
    word_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors

word_vectors = load_word_vectors(word2vec_file)

# 创建词向量矩阵，并转换为float32
embedding_dim = 256
embedding_matrix = np.zeros((len(word_to_idx), embedding_dim), dtype='float32')
for word, idx in word_to_idx.items():
    if word in word_vectors:
        embedding_matrix[idx] = word_vectors[word]

# 定义模型
import paddle.nn as nn

class LSTMModel(nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_size, context_size, embedding_matrix):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.embedding.weight.set_value(paddle.to_tensor(embedding_matrix))  # 加载预训练的词向量
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, direction='bidirectional')
        self.linear = nn.Linear(hidden_size * 2, vocab_size)  # 双向LSTM，乘以2

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        x = paddle.transpose(x, perm=[1, 0, 2])  # (seq_len, batch_size, embedding_dim)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (seq_len, batch_size, hidden_size * 2)
        last_time_step = lstm_out[-1]  # 取最后一个时间步
        out = self.linear(last_time_step)
        return out

# 加载已经训练好的模型
lstm_model = paddle.Model(LSTMModel(len(word_to_idx), embedding_dim, 256, 3, embedding_matrix))
lstm_model.load('lstm_modelSC')  # 加载模型权重

# 自定义预测函数，提取最后三个字并进行预测，排除指定词汇
exclude_words = {"我"}

def predict_next_word(context):
    # 使用正则表达式去除非中文字符，并提取最后三个汉字
    context = re.sub(r'[^\u4e00-\u9fa5]', '', context)
    
    if len(context) < 3:
        return ["请输入至少三个汉字"] * 5
    
    word = list(context[-3:])  # 提取最后三个字

    # 将输入单词转换为索引
    x_data = list(map(lambda w: word_to_idx.get(w, 0), word))
    x_data = paddle.to_tensor(np.array([x_data]))

    # 预测下一个单词
    predicts = lstm_model.network(x_data)
    predicts = predicts.numpy().tolist()[0]

    # 找出概率最大的预测词的索引
    top_10_idx = np.argsort(predicts)[-10:][::-1]  # 提取前10个词的索引作为候选

    # 获取前10个预测的单词，并过滤掉指定词汇
    predicted_words = [idx_to_word[idx] for idx in top_10_idx if idx_to_word[idx] not in exclude_words]

    # 去掉空格和重复的词
    final_predictions = []
    for word in predicted_words:
        if word != " " and word not in final_predictions and word != "<pad>":
            final_predictions.append(word)
        if len(final_predictions) == 5:  # 保证最多生成5个词
            break

    # 如果预测词数量不足5个，填充空值
    while len(final_predictions) < 5:
        final_predictions.append("")

    return final_predictions

def update_context_with_word(context, word):
    return context + word

with gr.Blocks(css="#Wrapper {border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); padding: 20px; background: url('https://cdn.pixabay.com/photo/2016/11/21/18/14/wall-1846965_1280.jpg') no-repeat center fixed; background-size: cover;} #Title {text-align: center; color: #333; font-size: 24px; margin-bottom: 20px;} #ImageContainer {display: flex; justify-content: center; margin-bottom: 20px;}") as demo:
    gr.HTML("<div id='Title'>人工智能诗词助手</div>")
    gr.HTML("""
        <div id="ImageContainer">
            <img src="https://img.picui.cn/free/2024/10/10/67075355195ed.png" alt="背景图片" style="width:40%; height:auto; object-fit:contain;">
        </div>
    """)
    
    
    with gr.Tab("诗词续写"):
        context_input = gr.Textbox(label="请输入任意长度的中文上下文", placeholder="例如：床前明月光")
        predict_button = gr.Button("预测下一个词")
        
        with gr.Row():
            word_button_1 = gr.Button("", elem_classes="btn")
            word_button_2 = gr.Button("", elem_classes="btn")
            word_button_3 = gr.Button("", elem_classes="btn")
            word_button_4 = gr.Button("", elem_classes="btn")
            word_button_5 = gr.Button("", elem_classes="btn")

        # 按钮交互
        predict_button.click(
            predict_next_word, 
            inputs=context_input, 
            outputs=[word_button_1, word_button_2, word_button_3, word_button_4, word_button_5]
        )

        # 每个按钮的点击事件将结果附加到上下文
        word_button_1.click(update_context_with_word, inputs=[context_input, word_button_1], outputs=context_input)
        word_button_2.click(update_context_with_word, inputs=[context_input, word_button_2], outputs=context_input)
        word_button_3.click(update_context_with_word, inputs=[context_input, word_button_3], outputs=context_input)
        word_button_4.click(update_context_with_word, inputs=[context_input, word_button_4], outputs=context_input)
        word_button_5.click(update_context_with_word, inputs=[context_input, word_button_5], outputs=context_input)
        
        predict_button.click(
          predict_next_word, 
          inputs=context_input, 
          outputs=[word_button_1, word_button_2, word_button_3, word_button_4, word_button_5]
        )

    with gr.Tab("意象替换"):
        with gr.Column():
            input_word = gr.Textbox(label="输入你想替换的意象")
            top_k = gr.Slider(minimum=1, maximum=10, step=1, label="显示意象数量", value=5)
            similar_words_output = gr.Textbox(label="最相似的意象")
            find_similar_btn = gr.Button("查找相似意象")
            # image_output = gr.Image(label="意象图片")

        find_similar_btn.click(
            find_similar_words,
            inputs=[input_word, top_k],
            outputs=[similar_words_output]
        )

    with gr.Tab("诗词生成图片"):
        text_input = gr.Textbox(label="输入诗词描述（可以给出你想要的风格哦(⊙_⊙)）", placeholder="请输入描述...")
        image_output = gr.Image(label="生成的图片")
        generate_btn = gr.Button("生成图片")

        generate_btn.click(
            generate_image,
            inputs=text_input,
            outputs=image_output
        )


if __name__ == "__main__":
    demo.launch()

