# HyperFlow H3 — Ascend NPU 适配

本文件只描述 NPU 适配层。原 CUDA 代码路径未改动，所有新增逻辑都在 `src/hyperflow_h3/npu.py` 与
`examples/common.py` 的设备抽象里；未设置 NPU 相关开关时行为与上游一致。

## 环境（已在本机装好）

| 项 | 版本 |
|---|---|
| Python | 3.12.16 |
| torch | 2.10.0+cpu（PyTorch CPU 源 / 上交镜像） |
| torch_npu | 2.10.0 |
| torchvision | 0.25.0+cpu |
| diffusers | 0.42.1 |
| transformers | 5.9.0 |
| CANN | 9.0.0 |
| 设备 | Ascend 910 ×2，单卡 61.27 GiB HBM |

安装命令（镜像站）：

```bash
cd /workspace/hyperflow
python3.12 -m venv .venv312
.venv312/bin/pip install -U pip -i https://mirrors.cloud.tencent.com/pypi/simple

# torch 必须来自 PyTorch CPU 源：PyPI 上的 torch 是 CUDA 构建，和 torch_npu 同时加载会 abort
.venv312/bin/pip install --no-deps "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" \
  --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cpu

.venv312/bin/pip install torch_npu==2.10.0 -i https://mirrors.cloud.tencent.com/pypi/simple

.venv312/bin/pip install "diffusers==0.42.1" "transformers==5.9.0" peft safetensors \
  huggingface_hub accelerate av pyyaml psutil -i https://mirrors.cloud.tencent.com/pypi/simple

.venv312/bin/pip install --no-deps -e .
```

> `torch_npu` 在 import 时需要 `yaml`、`psutil`，缺了会报 `ModuleNotFoundError: No module named 'yaml'`。

## 改了什么

### 1. 新增 `src/hyperflow_h3/npu.py`

| 函数 | 作用 |
|---|---|
| `is_npu_available()` | `torch_npu` 可导入且 `torch.npu.is_available()` 为真 |
| `default_device()` | 有 NPU 返回 `npu`，否则 `cuda`，否则 `cpu` |
| `resolve_device(x)` | `auto`/`None` → `default_device()`；显式值原样返回 |
| `device_module(dev)` | 返回 `torch.npu` / `torch.cuda` / `torch` |
| `make_generator(dev, seed)` | NPU 上创建 `torch.Generator(device="npu")`。**必须**：NPU 的 `randn` 拒绝 CPU generator，会报 `Expected a 'npu' device type for generator but found 'cpu'`。CUDA 上仍返回 CPU generator，保持逐位可复现 |
| `empty_cache` / `memory_allocated` / `max_memory_allocated` / `synchronize` / `stream` | 后端无关的显存与流操作 |
| `enable_npu_attention(model)` | 切换 diffusers 的 `_native_npu` 后端（内部走 `torch_npu.npu_fusion_attention`） |
| `enable_group_offloading(pipe, device)` | **核心**：把文本编码器和 DiT 按层/按 block 流式送卡（见下） |
| `install_device_aliases()` | 可选：`HYPERFLOW_NPU_ALIAS_CUDA=1` 时让 `tensor.is_cuda` 对 NPU 返回 True。默认关闭，因为它改变全局 torch 行为 |

### 2. 为什么必须 group offload（不是可选项）

Ascend 910 单卡 HBM 是 **61.27 GiB**：

| 组件 | bf16 大小 | 能否整块驻留 |
|---|---:|---|
| Qwen3-VL 文本编码器 | 62.13 GiB | ❌ 单独就超 |
| transformer / transformer_ref | 61.70 GiB | ❌ 装下也没有激活空间 |

`ComponentsManager.enable_auto_cpu_offload` 是**组件级**卸载：它先把整个组件 `.to(device)`，装不下就 OOM。
在 CUDA 的 141 GB H200 上没问题，在 Ascend 上必然失败。实测报错：

```text
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 252.00 MiB
(NPU 0; 61.27 GiB total capacity; 59.97 GiB already allocated; ...)
```

所以 NPU 路径改用 `apply_group_offloading`：

| 组件 | offload 级别 | 粒度 | 峰值显存 |
|---|---|---|---:|
| `text_encoder` | `leaf_level` | 单个权重张量 | 1.5 GiB 级 |
| `transformer` / `transformer_ref` | `block_level`，`num_blocks_per_group=1` | 一个 transformer block | 6–10 GiB |
| VAE / audio VAE | 不卸载（组件级） | 解码时整块上卡 | 11 GiB 级 |

权重在主机侧保留 pinned 副本，上载走第二条 NPU stream 与计算重叠（`use_stream=True, record_stream=True`），
已在本机实测 NPU 上可用。

### 3. `examples/common.py`

- `--device` 默认 `cuda` → `auto`：NPU → CUDA → CPU 解析；显式传 `cuda` 时行为与原来完全相同。
- `--gpus N` 多卡：设备前缀跟随后端（`npu:LOCAL_RANK`），分布式后端 `hccl`（CUDA 仍是 `nccl`）。
- `--attention-backend` 默认 `sdpa` → `auto`：Ascend 选 `_native_npu`，其他平台仍是 SDPA；新增 `npu` 选项。
- 新增 `--npu-offload/--no-npu-offload`，默认在 NPU 上开启，CUDA 上不影响。
- 新增 `--npu-offload-blocks N`：每个 group-offload 组包含几个 transformer block（默认 1）。
- 新增 `--tp [N]`：按权重做 tensor parallel；不带 N 时用所有 rank。
- `--sol-attn` 在 NPU 上直接报错退出（NVIDIA CUDA kernel，无 Ascend 版本）。
- 生成器改为 `npu_backend.make_generator(args.device, args.seed)`。
- 显存统计与 `empty_cache` 改走 `npu_backend`。
- `_pack_for_broadcast` 用 `device.type != "cpu"` 代替 `is_cuda`，让多卡广播在 NPU 上也生效。
- 多卡 rank 0 的 conditioner 单独加载文本编码器，也套同一套 group offload（TP 模式下同样需要，
  因为 TP 只切 DiT）。
- `SplitPipeline._warm_transformer` 跳过已 group offload 的 DiT（否则 `.to()` 会告警）。

### 4. 新增 `src/hyperflow_h3/tensor_parallel.py`

diffusers 的 TP 用在 MiniMax-H3 上有三个阻塞点，本模块逐个绕开：

| 阻塞点 | 处理 |
|---|---|
| `_SUPPORTED_TP_DEVICES` 没有 `npu` | `install_npu_tp_device()` 注册进去（DTensor + HCCL 实测可用） |
| `MiniMaxH3Transformer3DModel._tp_plan` 为 `None` | 本模块提供 `TP_PLAN` |
| `MiniMaxH3AttnProcessor` 用 `unflatten(-1, (attn.heads, -1))` | `TensorParallelAttnProcessor` 改用固定 `head_dim` |

以及一个更隐蔽的问题：

- **SwiGLU 无法切分**。`ff.net.0.proj` 是 `Linear(hidden, 2 * ffn_dim)`，`SwiGLU.forward` 用
  `chunk(2)` 分成 value/gate。
  - 普通 colwise：rank 0 拿到全部 value、rank 1 拿到全部 gate，各自算不出 `value * silu(gate)`。
  - `PackedColwiseParallel([1,1])`：本地分片是两段拼接（`value[0:16] + gate[0:16]`），但 DTensor 的
    `Shard(1)` 元数据声明的是连续分片 `[0:32]`，于是 `chunk(2)` 读到 `value[0:8]+value[8:16]` 与
    `gate[0:8]+gate[8:16]`，结果错误。

  处理：`SplitSwiGLU` 在最开始把融合层拆成 `proj_value` / `proj_gate` 两个独立 `Linear`，两者都是普通
  colwise，每张卡各持有 `inner/N` 列，`value * silu(gate)` 本地可算，后续 rowwise 的 `net.2` 正常
  all-reduce。实测 fp32 下相对误差 9.2e-07。

- **PEFT LoRA 无法切分**：`apply_tensor_parallel` 直接拒绝带 adapter 的模型；先切分又会因为 LoRA 矩阵是
  全尺寸、基座是 `1/N` 宽而在第一次 matmul 尺寸不符。处理：`merge_hyperflow_lora` 在切分前把 adapter
  合并进基座权重（推理上精确等价于 `y = (W + s·BA)x`），留下纯 `nn.Linear`。实测合并本身误差 4.5e-03
  （bf16），合并+TP 端到端两次运行逐像素相同。

### 5. 新增 `src/hyperflow_h3/npu_attention.py` 与 `npu_rmsnorm.py`（性能）

用 CANN profiler（`msprof` 与 `torch_npu.profiler`）在 1344×768 / 124 帧的一次 DiT forward 上取数。
先说结论：**单步内设备 100% 满载、零空隙**（26.05 s 窗口内没有一段 >0.5 ms 的 AI Core 空闲），所以没有
「填补空闲」的空间，只能换更快的算子。MatMul 已达这台机器的 bf16 上限（大 matmul 实测 287 TFLOPS，
profile 里 MatMulV3 约 300 TFLOPS），唯一显著低效的是 attention。

| 算子 | 占单步 | 实测效率 | 备注 |
|---|---:|---:|---|
| `npu_fusion_attention` BSND bf16（原状） | **63.4 %** | 123.6 TFLOPS | `aic_scalar_ratio` 0.87、`aic_mac_ratio` 0.38，标量管线打满 |
| MatMulV3 | 19.0 % | ~300 TFLOPS | 已到上限 |
| 向量算子（Mul/Add/Cast/Slice…） | 16 % | — | 其中 RMSNorm 可融合 |

两处独立改动：

**a) attention 按序列长度选 kernel**。`npu_prompt_flash_attention` 的 `BSH` 布局正好是
`(B, S, N, D)` 连续内存的另一种解读（无需转置、无需拷贝）。但在 768p（37K tokens）上的优势**不能外推**：
扫过序列长度后，两者各有胜负，而且长序列下差距极大（同机实测，bf16，head_dim 128，28 heads）：

| seq (tokens) | SDPA | prompt_flash | 胜者 |
|---:|---:|---:|---|
| 8,192 | **156.5 TF** | 136.6 TF | SDPA 1.15× |
| 37,718 | **158.3 TF** | 154.1 TF | SDPA 1.03× |
| 65,536 | 155.2 TF | **157.9 TF** | prompt_flash 1.02× |
| 109,480 | **161.4 TF** | 111.7 TF | SDPA 1.44× |
| 218,959 | **162.2 TF** | 94.0 TF | SDPA **1.72×** |

所以现在按长度自适应：只有 **60k–90k** 这个窄窗口走 `npu_prompt_flash_attention`，其余全走 PyTorch SDPA
（Ascend 上它经 `npu_fusion_attention` 落到同一个 CANN flash kernel，也是 CUDA 参考路径的选择，算术一致）。
两端都用同一条 `torch.nn.functional.scaled_dot_product_attention` 调用，只差一次 `transpose(1, 2)`，
219k 形状下该转置 2.6 ms、对 4.2 s 的 kernel 可忽略。

> 这条是踩过坑才定下来的：最初只按 768p 的数据选了 `prompt_flash`，在 1184×1760 / 362 帧（219k tokens）上
> 每步多花 **133 s**，8 步多 17.7 分钟。见「已知限制」里的长序列条目。

fp16 计算仍可通过 `compute_dtype` 指定，但默认 bf16：q/k/v 各 540 MB，三次 bf16→fp16 转换的开销超过
kernel 省下的时间（真实模型里 270 vs 260 ms/次）。

**b) RMSNorm 融合**。`F.rms_norm` 在 Ascend 上拆成 `Pow`/`ReduceMean`/`Rsqrt`/`Mul`/`Mul`；
`torch_npu.npu_rms_norm` 是单个融合算子。每次 forward 有 105 次 hidden-size 与 100 次 per-head 调用：

| 形状 | `nn.RMSNorm` | `npu_rms_norm` |
|---|---:|---:|
| hidden (5376)，105 次 | 6.37 ms | 0.84 ms |
| per-head (128)，100 次 | 8.35 ms | 0.90 ms |

约 1.33 s/step。精度与原来相同（对 fp32 参考都是 1.4e-03 mean relative error，即 bf16 舍入本身）。

两个开关默认在 NPU 上开启，CUDA/CPU 路径不受影响：

- `--npu-fast-attn/--no-npu-fast-attn`（等价 `--attention-backend npu_fast`）
- `--npu-fused-norm/--no-npu-fused-norm`

CP 下 attention 仍然生效：`fast_npu_attention` 把自带的 `forward_op` 交给 diffusers 的 Ulysses 包装器，
复用它的 all-to-all，不重写并行逻辑。

### 6. 推理前的初始化（profiler 分析）

用 `torch_npu.profiler` 在真实 2 MP 负载上逐阶段计时。一次冷启动的构成（3 次取中位数）：

| 阶段 | 耗时 | 说明 |
|---|---:|---|
| `load_components` | 3.2 s | 读 safetensors + 混合精度转换 |
| `load_hyperflow_lora` | 11.9 s | PEFT 逐模块注入 |
| `enable_group_offloading` | 21.7 s | 锁页 124 GiB + 注册 hook |
| 合计 | 34–41 s | |

逐项查过之后，**只有一项是真正可省的**：

**已采纳：LoRA 注入跳过随机初始化。** `LoraConfig(init_lora_weights=True)` 会对 316 个目标模块各跑一次
`kaiming_uniform_` + `zeros_`，但紧随其后的 `copy_` 会把每个 A/B 张量全部覆盖。改成
`init_lora_weights=False` 后中位数 **11.9 → 9.1 s**（隔离测量 3 次，每次新进程）。

正确性：A/B 两次端到端运行（同 seed，仅切换这一项）输出**逐像素相同**（maxdiff 0.0）。检查过权重文件里
`lora_B` 不是全零（是训练过的值），所以不存在“依赖零初始化”的语义。

**已否决（都实测过，不能采纳）：**

| 想法 | 实测 | 结论 |
|---|---|---|
| 关掉 `use_stream` 省掉锁页 | 初始化 38 → 22 s，但推理 37 → **111 s**（3×） | 否决，得不偿失 |
| 一次性锁页（`empty_like(pin_memory=True)` + `copy_`）替代 `cpu().pin_memory()` | 中位数 22.6 s vs 13.8 s | 否决，反而更慢：`pin_memory()` 是原地锁页，`empty_like+copy_` 多一次全量拷贝 |
| 多线程并发锁页 DiT + 文本编码器 | 17.6 s vs 18.7 s | 否决，`copy_` 持 GIL，线程不并行 |
| 多进程并发锁页 | 同载荷 3.6 s vs 1.1 s | 否决，更慢 |
| 关掉 VAE tiling | 27.7 s vs 21.9 s | 否决 |

剩下的 21.7 s 里约 14 s 是锁页 124 GiB（`stream` 必需，无法避免）、约 6 s 是 hook 注册，两者都是
diffusers 的固有成本。LoRA 那 9.1 s 是 316 × 25.6 ms 的逐模块替换，也压不动。

### 7. 不需要改的部分

- `diffusers` 0.42.1 自带 `_native_npu` attention 后端和 `TorchDeviceBackend`，`get_device()` 在 Ascend 上已能自动返回 `npu`。
- `diffusers` 的 group offloading 在 NPU 上可用（已实测 `use_stream=True/False` 均正常），`ComponentsManager.enable_auto_cpu_offload` 的内存查询走 `torch.npu.mem_get_info()`，可用。
- `safetensors` / `peft` / `transformers` 无 CUDA 硬编码阻塞点。

## 已知限制

| 项 | 说明 |
|---|---|
| 显存 | 单卡 61.27 GiB。bf16 文本编码器 62.13 GiB、transformer 61.70 GiB 都超过单卡容量，**必须**用 group offload（NPU 上默认开启） |
| `--sol-attn` | 不可用（CUDA-only kernel） |
| `--attention-backend fa3` | 不可用（FlashAttention-3 是 CUDA kernel）；Ascend 用 `npu` |
| `--tp` 与 `--npu-offload` | 互斥：TP 已把权重切成 `1/N`，再叠 group offload 会互相争夺参数的存放位置（diffusers 直接拒绝） |
| `--tp` 与 `--sol-attn` | 互斥：两者都要替换 attention processor |
| `--tp` 与 LoRA | 需要 `merge_hyperflow_lora`（CLI 自动执行）。合并后 `disable_hyperflow` / `unmerge` 不再可用，只适合推理 |
| `--tp` 性能 | 本机 2 卡 TP 比 2 卡 CP 慢（768p：175.9 s vs 135.3 s），且显存占用更高（55.8 vs 9.1 GiB）。少步数模型里 all-reduce 占比太高 |
| 超长序列的 attention | tokens 超过约 10 万后 `npu_prompt_flash_attention` 的效率会掉到 ~95 TFLOPS，而 SDPA 仍保持 162。已按长度自适应规避（60k–90k 之外全走 SDPA）。但 attention 本身仍是 O(S²)：1184×1760 / 362 帧 = 21.9 万 tokens，单步 390 s 里 360 s 是 attention，这是该画布的物理成本，不是可优化的开销 |
| `torch.accelerator` 显存 API | `torch.accelerator.memory_allocated()` 在 torch_npu 2.10 会触发 allocator 断言，因此 `npu.py` 直接调用 `torch.npu.*` |
| 逐位一致 | TP 自身可复现（同机同 seed 逐像素相同）；TP 与单卡之间、NPU 与 CUDA 之间不保证逐位相同 |

## 验证结果（本机实测）

`.venv312/bin/python -m pytest -q` → **237 passed**（含 NPU、TP、WebUI 与性能开关用例，NPU 不可用时自动 skip）。

### 640×384 / 124 帧 / 8 步

| 任务 | 设备 | 耗时 | 峰值显存 | 产物 |
|---|---|---:|---:|---|
| `fl2va`（首帧） | 1× NPU，group offload | 39.2 s | 6.4 GiB | `outputs/npu_fl2va_smoke.mp4` |
| `ref2va`（图像参考） | 1× NPU，group offload | 119.9 s | 9.4 GiB | `outputs/npu_ref2va_smoke.mp4` |
| `t2va` | 2× NPU，**CP**（HCCL） | 28.3 s | 6.2 GiB | `outputs/npu_t2va_2card_v2.mp4` |
| `fl2va` | 2× NPU，**TP** | 64.1–83.4 s | 52.0 GiB | `outputs/tp2_fixed_640p.mp4` |

### 1344×768 / 124 帧（5.2 s）/ 8 步

| 并行方式 | 优化前 | 优化后 | 峰值显存 | 产物 |
|---|---:|---:|---:|---|
| 单卡 group offload（blocks=1） | 237.6 s | **203.9 s** | 12.7 GiB | `outputs/npu_fl2va_768p_1card.mp4` |
| 单卡 group offload（blocks=4） | 238.5 s | — | 12.7 GiB | `outputs/npu_fl2va_768p_offload4.mp4` |
| 2 卡 **CP**（Ulysses 按 token 切） | 135.3 s | **123.2 s** | 9.1 GiB | `outputs/npu_fl2va_768p_cp2.mp4` |
| 2 卡 **TP**（按权重切） | 175.9 s | — | 55.8 GiB | `outputs/npu_fl2va_768p_tp2_fixed.mp4` |

「优化后」为启用 `npu_fast` attention + 融合 RMSNorm 后的 `pipe()` 总时长（同机同 seed 重测）：

| 阶段 | 单卡 | 2 卡 CP |
|---|---:|---:|
| denoise（8 步） | 211.8 → 177.7 s | 108.8 → 96.3 s |
| 视频解码 | 21.8 → 21.8 s | 21.8 → 22.1 s |
| 合计 | 238.1 → 203.9 s（-14.4 %） | 136.1 → 123.2 s（-9.5 %） |

视频解码没有变快：VAE 默认 `use_tiling=True`（tile 256 px），tile 调大能到 15.9 s（448 px）但会改变
tiling 混合边界、画面与本仓库既有产物不一致，因此保持默认。CP 下每个 rank 都会解码同一份 latent，
这部分不是并行瓶颈，属于后续可做的工作（只让 rank 0 解码）。

结论：

- **768p 首选 CP**：123.2 s（优化前 135.3 s），相对单卡 203.9 s 是 1.65×，显存还降了（9.1 vs 12.7 GiB）。
- **TP 在这台机器上更慢**（175.9 s，未做上面的两项优化）：每层 attention/FFN 边界都要 all-reduce，Ascend 910 的 HCCL
  带宽不足以摊薄 8 步少步数带来的通信占比；而且单卡显存吃满 55.8 GiB（权重只切了一半，激活没切）。
- `--npu-offload-blocks 4` 无收益：单卡瓶颈在算力而非 H2D 拷贝。
- 产物均为 H.264 + AAC 32 kHz 双声道，124 帧 @ 24 fps，画面语义正确。

### TP 的数值一致性

| 场景 | 相对误差 |
|---|---:|
| 2 层 tiny 模型，fp32，TP vs 单卡 | 9.2e-07 |
| 2 层 tiny 模型，bf16 + LoRA 合并，TP vs 单卡 | 5.4e-03 |
| 真实模型，fp32，LoRA + 合并 + TP vs 单卡 | 8.6e-07 |
| 端到端 640p，TP 两次重跑 | **逐像素相同** |
| 端到端 640p/768p，TP vs 单卡同 seed | 画面相关系数 0.83–0.94 |

TP 自身是确定性的（两次逐像素一致），与单卡的差异来自 all-reduce 改变了 bf16 归约顺序 ——
这与 CUDA 上「不同 GPU 数产生不同 clip」是同一类现象（官方 README 也写明
`Reproducibility: same seed, GPU architecture, GPU count and attention backend`）。

## 验证命令

```bash
cd /workspace/hyperflow
.venv312/bin/python -m pytest -q          # 237 passed（CPU 单测 + NPU 用例）

# 设备与算子
.venv312/bin/python -c "
import torch, torch_npu
print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())
"

# 端到端
export HF_HUB_OFFLINE=1
.venv312/bin/hyperflow-h3-fl2va --prompt "a red ball bouncing on a wooden table" \
  --model  /workspace/hyperflow/models/MiniMax-H3 \
  --weights /workspace/hyperflow/models/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors \
  --gpus 1 --device npu --num-frames 124 --width 640 --height 384 \
  --output /workspace/hyperflow/outputs/npu_fl2va.mp4
```

`--gpus 1` 是官方单卡 recipe；NPU 上会自动叠加 group offload，无需额外参数。

