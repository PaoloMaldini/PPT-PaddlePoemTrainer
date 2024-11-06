from flask import Flask, request, jsonify
import paddle
import numpy as np
# (导入其他所需库和模型)

app = Flask(__name__)
def load_model(model, model_path):
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    return model
# 加载模型的代码
model = load_model(model, best_model_path) # 替换为你的模型加载逻辑

# 定义预测接口
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("input_text")
    result = predict_next_word(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
