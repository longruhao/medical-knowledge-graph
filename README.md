```markdown
# 医疗知识图谱构建系统

基于深度学习的医疗领域命名实体识别（NER）和关系抽取（RE）系统，用于从电子病历中自动提取医疗实体及其关系，构建医疗知识图谱。

## 📋 项目简介

本项目实现了两种主流的 NLP 任务：

1. **命名实体识别（NER）**：从医疗文本中识别疾病、症状、治疗、检查、身体部位等实体
2. **关系抽取（RE）**：识别实体之间的关系，如"疾病-症状"、"疾病-治疗"等

系统支持多种模型架构，包括 BiLSTM-CRF、CasRel 等先进算法，并提供 Flask Web 服务接口。

## ✨ 主要特性

- 🎯 **多模型支持**：BiLSTM、BiLSTM-CRF、CasRel、BiLSTM-Attention
- 🏥 **医疗领域专用**：针对电子病历优化的实体识别和关系抽取
- 🌐 **Web 服务接口**：提供 RESTful API 和可视化界面
- 📊 **知识图谱存储**：支持 Neo4j 图数据库存储和查询
- 🔧 **模块化设计**：清晰的代码结构，易于扩展和维护
- 📈 **性能评估**：完整的训练、验证、测试流程

## 🏗️ 项目结构

```

knowledge_graph/
├── ner_bi_lstm_crf/              # 命名实体识别模块
│   ├── data/                     # 数据集
│   ├── model/                    # 模型定义
│   │   ├── BiLSTM.py            # BiLSTM 模型
│   │   └── BiLSTM_CRF.py        # BiLSTM-CRF 模型
│   ├── utils/                    # 工具函数
│   ├── save_model/              # 保存的模型
│   ├── train.py                 # 训练脚本
│   ├── predict.py               # 预测脚本
│   ├── app.py                   # Web 应用
│   └── config.py                # 配置文件
│
├── relation_extraction/          # 关系抽取模块
│   ├── re_casrel/               # CasRel 模型
│   │   ├── model/               # CasRel 模型定义
│   │   ├── utils/               # 数据处理工具
│   │   ├── train.py             # 训练脚本
│   │   ├── predict.py           # 预测脚本
│   │   ├── map_display.py       # Neo4j 图谱展示
│   │   └── config.py            # 配置文件
│   ├── re_lstm_attention/       # BiLSTM-Attention 模型
│   │   ├── model.py             # 模型定义
│   │   ├── train.py             # 训练脚本
│   │   └── predict.py           # 预测脚本
│   └── re_rule_based/           # 基于规则的方法
│       └── re_rule.py           # 规则抽取
│
├── tools/                        # 辅助工具
│   ├── data_process.py          # 数据预处理
│   └── experiments.py           # 实验代码
│
├── requirements.txt              # 依赖包
└── README.md                     # 项目说明
```
## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (可选，用于 GPU 加速)

### 安装步骤

1. **克隆项目**
```
bash
git clone https://github.com/your-username/knowledge_graph.git
cd knowledge_graph
```
2. **安装依赖**
```
bash
pip install -r requirements.txt
```
3. **下载预训练模型**（仅 CasRel 需要）

首次运行 CasRel 模型时，会自动从 HuggingFace 下载 `bert-base-chinese` 模型（约 400MB）。

如果网络较慢，可以设置镜像：
```
bash
# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"

# Linux/Mac
export HF_ENDPOINT=https://hf-mirror.com
```
## 📊 数据集

### NER 数据集

位于 `ner_bi_lstm_crf/data_origin/`，包含四个类别的医疗文本：
- 一般项目
- 病史特点
- 诊疗经过
- 出院情况

数据格式采用 BIO 标注体系：
```

咳    B-SIGNS
嗽    I-SIGNS
，    O
发    B-SIGNS
热    I-SIGNS
```
### RE 数据集

位于 `relation_extraction/re_casrel/data/`，包含 18 种关系类型：
- 伴随症状、治疗方法、检查项目、发病部位
- 药物治疗、手术治疗、并发症、病因
- 预防措施、诊断方法、预后情况等

## 🎯 使用方法

### 1. 命名实体识别（NER）

#### 训练模型

```
bash
cd ner_bi_lstm_crf

# 使用 BiLSTM-CRF 模型（推荐）
python train.py

# 或使用 BiLSTM 模型
# 修改 config.py 中的 self.model = 'BiLSTM'
```
训练完成后，最佳模型保存在 `save_model/bilstm_crf_best.pth`

#### 预测实体

```
python
from predict import model2test

text = "患者缘于3天前进食油腻食物后出现腹痛伴腹胀"
result = model2test(text)
print(result)
# 输出: {'腹痛': 'SIGNS', '腹胀': 'SIGNS'}
```
#### 启动 Web 服务

```
bash
# 启动 NER 服务
python solt_app.py

# 访问 http://127.0.0.1:6002/service/api/medical_ner
```
API 示例：
```
bash
curl -X POST http://127.0.0.1:6002/service/api/medical_ner \
  -H "Content-Type: application/json" \
  -d '{"text": "患者患有糖尿病和高血压"}'
```
### 2. 关系抽取（RE）

#### 训练 CasRel 模型

```
bash
cd relation_extraction/re_casrel

# 开始训练
python train.py
```
训练过程中会定期在验证集上评估，并保存最佳模型到 `save_model/best_f1.pth`

#### 预测关系

```
python
from predict import load_model, model2predict

# 加载模型
model = load_model('save_model/best_f1.pth')

# 预测
text = "《秋天的眼泪》是孟庭苇演唱的歌曲"
result = model2predict(text, model)
print(result)
# 输出: {'text': '...', 'spo_list': [{'subject': '秋天的眼泪', 
#          'predicate': '歌手', 'object': '孟庭苇'}]}
```
#### 启动 Web 服务

```
bash
python flask_web.py

# 访问 http://127.0.0.1:5007/predict/
```
#### 构建知识图谱

```
python
from map_display import load_file_create_map, use_neo4j2search

# 创建图谱（需要先运行 ready_data() 生成预测数据）
load_file_create_map()

# 查询图谱
use_neo4j2search()
```
## ⚙️ 配置说明

### NER 配置 (`ner_bi_lstm_crf/config.py`)

```
python
class Config():
    # 设备选择
    device = 'cuda'  # 或 'cpu'
    
    # 模型参数
    embedding_dim = 300
    hidden_dim = 256
    dropout = 0.2
    
    # 训练参数
    model = 'BiLSTM_CRF'  # 或 'BiLSTM'
    epochs = 20
    batch_size = 16
    lr = 2e-4
```
### RE 配置 (`relation_extraction/re_casrel/config.py`)

```
python
class Config(object):
    # BERT 模型
    bert_model_name = 'bert-base-chinese'
    
    # 训练参数
    epochs = 10
    batch_size = 16
    learning_rate = 1e-5
```
## 📈 模型性能

### NER 模型对比

| 模型 | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| BiLSTM | ~0.85 | ~0.83 | ~0.84 |
| BiLSTM-CRF | ~0.90 | ~0.88 | ~0.89 |

### RE 模型性能

CasRel 模型在医疗关系抽取任务上的表现：
- Subject F1: ~0.85
- Triple F1: ~0.78

*注：具体性能取决于数据集和训练参数*

## 🔧 常见问题

### 1. BERT 模型下载失败

**问题**：首次运行时 BERT 模型下载超时

**解决方案**：
```
bash
# 设置国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型后放置到指定目录
```
### 2. CUDA 内存不足

**问题**：训练时显存溢出

**解决方案**：
- 减小 `batch_size`
- 使用梯度累积
- 启用混合精度训练

### 3. 数据文件为空

**问题**：`train.json` 或 `test.json` 文件大小为 0

**解决方案**：
系统会自动从训练集划分验证集，无需单独的 `dev.json` 文件。如需生成 JSON 格式数据，可使用 `relation_extraction/data_origin/create.py`

### 4. 导入错误

**问题**：`ModuleNotFoundError`

**解决方案**：
```
bash
# 确保在项目根目录运行
cd knowledge_graph

# 重新安装依赖
pip install -r requirements.txt
```
## 📚 技术栈

- **深度学习框架**：PyTorch 2.0+
- **预训练模型**：Transformers (BERT)
- **Web 框架**：Flask
- **数据处理**：NumPy, Pandas
- **中文分词**：Jieba
- **图数据库**：Neo4j, py2neo
- **性能测试**：Locust
- **可视化**：Matplotlib

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目仅供学习和研究使用。

## 📧 联系方式

如有问题或建议，欢迎通过 GitHub Issues 联系。

## 🙏 致谢

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Neo4j](https://neo4j.com/)

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**
```
