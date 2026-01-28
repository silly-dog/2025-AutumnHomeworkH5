1. # 多模态情感分类项目

   本项目基于成对的文本（`.txt`）与图像（`.jpg`）样本，以多模态融合方式预测三类情绪：`negative`、`neutral`、`positive`。

   ## 核心结构

   - Code.py：主训练入口，优先尝试微调 CLIP（`openai/clip-vit-base-patch32`），若下载或执行失败则退回到自研轻量融合模型。后者包含自定义分词、文本/图像编码器以及融合分类头；两种流程都会输出 `submission.csv`。
   - Ablation.py：精简的消融脚本，加载 CLIP checkpoint，比较融合、文本、图像三种模型在验证集上的表现，帮助分析模态贡献。
   - data/：存放所有 `guid` 对应的 `.txt` + `.jpg` 配对样本，训练/测试时样本必须同时存在文本与图像文件。
   - train.txt / test_without_label.txt：CSV 格式的元信息，包含 `guid` 与 `tag` 列（测试集缺少标签）。

   ## 依赖要求

   - Python 3.10 以上，需安装 `torch`、`transformers`、`scikit-learn`、`pandas`、`Pillow`。
   - 建议在支持 CUDA 的 GPU 环境运行主流程；CLIP 会自动在 CUDA 上启用 float16，回退模型即使在 CPU 上也能执行。
   - 若希望微调 CLIP 全量参数、而非冻结 backbone，请确保 `cuda` 设备可用。
   - 更详细的依赖可以见 [requirements.txt](requirements.txt)。

   ## 执行主流程（Code.py）

   1. 将所有样本文件放入 `data/`，确保 `train.txt` / `test_without_label.txt` 中的 `guid` 与文件一一对应。
   2. 安装依赖：`pip install torch torchvision transformers scikit-learn pandas pillow`。
   3. 运行 `python Code.py`：
      - 校验每个 `guid` 都对应文本与图像。
      - 将训练数据按标签进行分层切割，取 10% 作为验证集。
      - 尝试加载 CLIP 并训练，若失败则自动调用轻量模型。
      - 用最终模型预测测试集并输出 `submission.csv`。

   轻量回退模型自带图像增强和文本编码，避免依赖外部大模型与 TorchVision，可在资源受限环境下运行。

   ## 消融分析（Ablation.py）

   运行 `python Ablation.py` 以：

   - 自动下载（或复用）`openai/clip-vit-base-patch32`，避免重复载入。
   - 读取 `train.txt`，按 `label` 分层留出 10% 验证集。
   - 依次评估多模态融合、纯文本、纯图像三种 head，在验证集上输出准确率与宏平均 F1。 

   脚本会在本地缓存 `clip_ablation.pth`，后续无需再次训练即可评估。

   ## 结果与产物

   - `submission.csv`：预测结果，包含 `guid` 与模型预测的 `tag`。
   - `clip_ablation.pth`：`Ablation.py` 保存的 CLIP 融合模型权重，用于快速加载。

   ## 建议下一步

   1. 根据自身硬件调节 `Code.py` 中的超参（`EPOCHS`、`IMG_SIZE`、`MAX_LEN` 等）以提升效果。
   2. 拓展 `Ablation.py`，记录各类指标或可视化模态特征。
   3. 生成 `submission.csv` 后务必检查是否包含所有测试 `guid` 并符合提交格式。