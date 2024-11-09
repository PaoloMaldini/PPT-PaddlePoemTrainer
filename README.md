
# 人工智能诗词助手

本项目为一个人工智能诗词助手应用，基于PaddlePaddle和Gradio，实现了以下功能：
1. 诗词续写：根据上下文预测诗词的下一个字。
2. 意象替换：查找与输入词相似的意象。
3. 诗词生成图片：根据输入的诗词生成相应的图片。

## 项目依赖

- Python 3.x
- PaddlePaddle
- PaddleNLP
- Gradio
- 其他依赖：requests, numpy

## 快速开始

1. **克隆项目**

   ```bash
   git clone https://github.com/PaoloMaldini/PPT-PaddlePoemTrainer.git
   cd PPT-PaddlePoemTrainer.git
   ```



3. **准备模型文件**

   - 将 `best_model_epoch_1.pdparams` 放置于项目根目录。
   - 确保 `words.txt`、`word_vec_SC.txt`、`SC_word_to_idx.pkl` 和 `SC_idx_to_word.pkl` 文件已放置于项目根目录。

4. **运行项目**

   ```bash
   python Gradio.py
   ```

## 功能介绍

### 1. 诗词续写
- 输入任意长度的中文上下文，预测下一个词。
- 通过点击预测词更新输入上下文，实现诗句自动续写。

### 2. 意象替换
- 输入关键词，系统查找最相似的意象并返回。

### 3. 诗词生成图片
- 输入诗词描述，生成相应风格的图像。

## 代码说明

- `ErnieForSequenceRegression`：定义了基于Ernie模型的回归层，用于序列相似性评估。
- `LSTMModel`：加载预训练的LSTM模型，并结合词嵌入实现上下文分析。
- `precompute_embeddings`：预先计算词汇的词嵌入，供意象替换功能使用。
- `generate_image`：与百度API集成，基于输入诗词生成图像。

## 文件结构

- `Gradio.py`：项目主文件，包含Gradio界面和功能实现。
- `best_model_epoch_1.pdparams`：训练好的模型文件。
- `word_vec_SC.txt`，`SC_word_to_idx.pkl`，`SC_idx_to_word.pkl`：词汇文件和映射。

## 示例

### 诗词续写
- 输入：“床前明月光”，输出下一个可能的词汇，并可以继续点击生成更多词汇。

### 意象替换
- 输入词语“明月”，将显示最相似的词。

### 诗词生成图片
- 输入描述“夜色中的明月”，自动生成一张相关图片。

## 注意事项

- 确保有百度API权限，并在`API_KEY`与`SECRET_KEY`中填入有效的百度API密钥。
- 生成图片时可能需要一定时间，请耐心等待。

## 许可
MIT License

---

