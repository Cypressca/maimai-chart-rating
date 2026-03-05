# maimai 谱面定数预测

## 0. 双击运行（源码模式）
现在 `maimai_const_predictor.py` 支持无参数交互模式，可双击运行（会弹出终端菜单）。

## 1. 安装依赖
在当前目录执行（建议使用 `.venv`）：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. 训练模型

```powershell
.\.venv\Scripts\python.exe maimai_const_predictor.py train --root . --output maimai_level_model.pkl
```

训练会自动递归读取所有 `maidata.txt`，从 `&lv_n` 提取定数标签、从 `&inote_n` 抽取谱面特征。

## 2.1 训练原理
本项目把“定数预测”建模为**监督学习回归问题**，核心流程如下：

1. **样本构建（标签）**
  - 遍历目录下所有 `maidata.txt`。
  - 对每个难度 `n`，若同时存在 `&lv_n`（定数）和 `&inote_n`（谱面正文），则生成一条训练样本。
  - 标签 `y` 即该难度对应的定数（浮点数）。

2. **特征工程（输入）**
  - 从谱面 token 中统计结构特征：
    - 总 token 数、hold 数、slide 数、break 数、ex 数、touch 数
    - token 长度统计（最大值/均值/标准差）
    - 数字密度、各类音符占比（ratio）
  - 同时加入全局信息：`wholebpm`、机台类型（`DX/SD`）、难度编号（`difficulty_index`）。
  - 将特征字典用 `DictVectorizer` 转成数值向量。

3. **模型训练**
  - 使用 `RandomForestRegressor`（500 棵树）训练。
  - 按 `train_test_split` 将样本拆分为训练集和测试集（默认 8:2）。
  - 由于随机森林能拟合非线性关系，适合处理谱面统计特征到定数之间的复杂映射。

4. **评估与保存**
  - 评估指标：`MAE`、`RMSE`、`R²`（训练集与测试集都会输出）。
  - 将模型与向量器一起保存到 `maimai_level_model.pkl`，预测时复用同一套特征变换。

5. **推理阶段**
  - 输入一个 `maidata.txt + difficulty`。
  - 用与训练完全相同的特征提取和向量化流程，得到预测定数。

> 说明：当前方案属于“基于统计特征的传统机器学习”。优点是快、可解释、易部署；若后续追求更高精度，可继续扩展特征或尝试 GBDT/XGBoost/深度模型。

## 3. 预测单个谱面定数

```powershell
.\.venv\Scripts\python.exe maimai_const_predictor.py predict --model maimai_level_model.pkl --maidata "24.maimai DX PRiSM\11789_ABSTRUSEDILEMMA_DX\maidata.txt" --difficulty 5
```

## 4. 参数说明
- `train`
  - `--root`：训练数据根目录（默认当前目录）
  - `--output`：模型输出路径（默认 `maimai_level_model.pkl`）
  - `--test-size`：测试集比例（默认 `0.2`）
  - `--random-state`：随机种子（默认 `42`）
- `predict`
  - `--model`：模型文件路径
  - `--maidata`：目标谱面文件路径
  - `--difficulty`：难度编号（通常 `2~5`）

## 5. 当前一次训练参考指标
- 样本数：6147
- 测试集 MAE：0.533
- 测试集 RMSE：0.709
- 测试集 R²：0.959

## 6. 打包为可迁移 EXE（推荐）
生成后可在其他 Windows 机器直接双击运行，无需额外安装 Python/插件。

```powershell
powershell -ExecutionPolicy Bypass -File .\build_portable.ps1
```

打包完成后使用：
- `portable_release\maimai_const_predictor_portable.exe`（可直接双击）
- 或双击 `portable_release\start_portable.bat`
