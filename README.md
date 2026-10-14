# OpenVLA Black-box Uncertainty

在 OpenVLA 与 LIBERO / LIBERO-Reflect 上评测 **Deg、EigV、Ecc、KLE、SPUQ**。仓库包含独立的推理、评分、审计、多卡启动和断点续跑代码。

五种方法的原始实现面向问答。本项目保留其图评分和加权汇总公式，将文本关系改为六维归一化动作的 RBF 相似度与夹爪动作一致性。SPUQ 使用四条保留任务含义的指令改写。它们是动作空间适配，并非直接复用问答实验设置。

默认执行完整动作轨迹，在第 `0, 20, 40, ...` 步计算分数，并取轨迹均值。评分间隔是本项目参数；改成每步评分可传 `--stride 1`。辅助采样不执行，机器人始终执行原始 greedy 动作。

## 新服务器安装

已验证的环境为 Linux、Python 3.10、PyTorch 2.6 / CUDA 12.4、Transformers 4.40.1、MuJoCo 3.9.0、robosuite 1.4.1，以及 A100 80GB。先确保服务器的 NVIDIA 驱动和 EGL 可用。

```bash
git clone https://github.com/dykzn/uncertainty.git
cd uncertainty
bash scripts/setup_env.sh
```

私有仓库需要先完成 GitHub 登录。环境安装和下载脚本禁用代理。显存较小的 GPU 建议从每卡一份评测开始；单份模型和解码缓存通常需要约 18–20GiB。

## 数据与权重

```bash
.venv/bin/python scripts/download_assets.py
.venv/bin/python scripts/download_checkpoints.py --suite all
.venv/bin/python scripts/preflight.py --weights
```

数据下载固定在 LIBERO-Reflect 的 `3b8e6eb2501969b441fdaaf9da32c14ba77e4bfd` 版本，同时提供标准场景和 `swap` 场景。四个套件的 80 个 BDDL 文件与 80 个初始状态文件会与保存的 SHA256 校验；无需重新生成场景。

四套权重分别来自 `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`。版本写在 `config/checkpoints.json`。代码使用同一份本地模型实现，支持返回动作 token，并校验权重完整性。模型权重和约 1.1GB 的数据资产在服务器单独下载，Git 仓库只保存代码与小型配置。

如果已有文件，可以指定路径：

```bash
export OPENVLA_CHECKPOINT_ROOT=/path/to/checkpoints
export LIBERO_REFLECT_ROOT=/path/to/LIBERO-REFLECT
# 可选：标准场景与 Reflect 不在同一个下载目录时指定标准数据根目录。
# export LIBERO_DATA_ROOT=/path/to/LIBERO/libero/libero
.venv/bin/python scripts/preflight.py --weights
```

已有权重应保留 `.hfd/repo_metadata.json`。对于缺少这份元数据的 Hugging Face 下载目录，可以运行下载脚本；它会复用尺寸正确的文件并创建元数据。

SPUQ 的任务改写及 MiniLM 相似度权重已预计算并包含在 `assets/*_instructions.json` 中，正常评测不需要再加载文本编码器。

修改任务改写后可重新计算相似度权重：

```bash
.venv/bin/python scripts/download_checkpoints.py --embedding
.venv/bin/python prepare_formal_assets.py
```

## 启动四套完整评测

每套 10 个任务 × 50 个初始状态 × 标准/Reflect 两种场景，总计 **4000 条轨迹**。每条最多执行 520 个策略动作，初始化等待动作另计。

```bash
# 双卡 A100 80GB：每卡最多四份模型，比较并发吞吐后选择运行配置。
.venv/bin/python start_formal.py --gpus 0,1 --max-workers 4 --stride 20

# 单卡：默认每卡一份模型，其余套件自动排队。
.venv/bin/python start_formal.py --gpus 0 --max-workers 1 --stride 20
```

GPU 参数对应 `nvidia-smi` 的编号。程序按 GPU UUID 匹配 EGL 渲染设备，均分初始状态，并避免重复轨迹。多卡并发不会按进程数量线性提速；程序会测量实际动作吞吐量。

启动后进程在后台运行。启动命令打印结果目录和进程号；关闭终端不会结束评测。输出位置默认是 `outputs/formal/<时间>/`：

- `status.json`：进度、GPU 队列、并发速度。
- `metrics.csv`：每套标准、Reflect、混合场景的 AUROC、AUPR（AP）、FPR@95，单位为百分比。
- `summary.json`：成功/失败数量与完整指标；仅有单一类别时指标为 `null`。
- `<suite>/shard*/worker.log`：推理日志。
- `<suite>/shard*/artifacts/`：评分时刻的图像、token、动作、相似度矩阵与分数。
- `finalization.json`：所有轨迹完成并通过独立审计后生成。

AUROC/AUPR 以成功为正类。轨迹均值属于事后评分，不是仅凭起始图像的成功预测。

先做小规模验证也可以：

```bash
.venv/bin/python start_formal.py --gpus 0 --max-workers 1 \
  --suites libero_spatial --task-end 1 --initial-count 1 --stride 20
```

## 断点续跑与迁移结果目录

本仓库生成的运行记录使用相对路径。先停止旧机器上的对应评测，将整个批次目录复制到新机器；保证代码版本、模型版本、任务配置一致，再执行：

```bash
.venv/bin/python start_formal.py --resume /path/to/copied-batch \
  --gpus 0,1 --max-workers 4
```

续跑保持原批次的评分间隔、分片数量和初始状态划分；GPU 编号可以变化。已完成轨迹保留，未完成轨迹重新执行。源文件与运行配置的哈希会在续跑时校验。早期实验脚本生成的旧格式记录不使用本仓库的迁移格式。

## 加速与校验

CUDA Graphs 捕获 eager 解码，不切换注意力实现。采样保持完整词表的 softmax / multinomial 和随机数状态；仅在同一张观测图像内缓存视觉特征。每个评分时刻校验 greedy token；新长度与定期采样检查 token 和 RNG，一旦不一致就回退到原始推理。

参考验证在 A100 上比较过 72 组采样、504 个 token，采样 token、随机数状态和五种分数一致。新服务器还会执行启动和运行时校验。独立仿真重跑可能产生不同轨迹，参考记录不代表新机器的每条轨迹都与原机器逐步相同。

```bash
.venv/bin/python check_scores.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python audit_run.py outputs/formal/<批次>/<suite>/shard0
```

评分测试直接提取作者核心函数，在 102 组矩阵上比较。完成一个分片后自动审计动作解码、样本共享、查询覆盖、分数与轨迹汇总。

源码来源和复制范围记录在 `validation/export_provenance.json`；第三方版权说明见 `THIRD_PARTY_NOTICES.md`。
