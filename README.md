# Visual Anagrams 视觉错觉生成器

基于扩散模型的多视角光学错觉图像生成项目。

## 环境配置

```bash
pip install -r requirements.txt
pip install git+https://github.com/dangeng/visual_anagrams.git
```

## 使用前准备

1. 注册 [HuggingFace](https://huggingface.co/join) 账号
2. 在 [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) 接受模型使用条款
3. 获取 [Access Token](https://huggingface.co/settings/tokens)

## 运行

```bash
python main.py --token YOUR_HF_TOKEN --prompt1 "painting of a mountain" --prompt2 "painting of a cat" --view rotate_cw --animate
```

## 参数说明

| 参数 | 说明 |
|------|------|
| --token | HuggingFace访问令牌 |
| --prompt1 | 第一个视角的文本提示 |
| --prompt2 | 第二个视角的文本提示 |
| --view | 变换类型 (rotate_180, rotate_cw, rotate_ccw, flip等) |
| --output | 输出路径前缀 |
| --animate | 生成动画视频 |

## 项目结构

```
├── main.py          # 主入口
├── generate.py      # 生成器核心逻辑
├── config.py        # 配置参数
├── utils.py         # 工具函数
└── requirements.txt # 依赖列表
```

