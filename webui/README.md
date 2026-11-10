# HyperFlow H3 WebUI

Ascend NPU 上的浏览器前端。只用标准库（`http.server` + `email` 解析 multipart），不引入额外依赖。

```bash
./scripts/launch_webui.sh            # http://0.0.0.0:3000
HOST=127.0.0.1 PORT=8080 ./scripts/launch_webui.sh
```

环境变量：

| 变量 | 作用 |
|---|---|
| `HYPERFLOW_WEBUI_MODEL` | 覆盖底座快照路径（默认自动探测 `models/MiniMax-H3`） |
| `HYPERFLOW_WEBUI_WEIGHTS` | 覆盖 LoRA 文件（默认自动探测 `models/hyperflow/*.safetensors`） |
| `HYPERFLOW_WEBUI_RUNS` | 运行目录（默认 `webui/runs`） |
| `HYPERFLOW_WEBUI_ACCESS_LOG=1` | 打开访问日志 |

## 时长：15 s 需要额外放宽上限

`MiniMaxH3ModularPipeline.max_duration` 是 **15.0 s**，而且 diffusers 是对**对齐后**的帧数做检查的。360 帧
（15 s）会向上对齐到 362 帧 = 15.083 s，因此**原版会被拒绝**：

```python
duration = align_num_frames(num_frames) / fps      # 362 / 24 = 15.083
if not min_duration <= duration <= max_duration:   # 15.083 > 15.0 -> ValueError
```

所以 `hyperflow_blocks` 配套一个 `admit_max_frames(pipe)`，把上限抬到 362 帧 —— 与
[`hyperflow-h3-flash`](../hyperflow-h3-flash) 里的同名函数做法一致（改实例的类，因为 `max_duration` 是只读
property）。CLI 在 `init_pipeline` 之后立刻调用它，单卡与多卡路径都覆盖。

`webui/geometry.py` 的校验也因此改成检查**对齐后的帧数**，而不是用户输入的秒数：

| 输入 | 结果 |
|---|---|
| 15.0 s | ✅ 362 帧 / 15.083 s |
| **15.04 s** | ✅ 362 帧 / 15.083 s |
| 15.1 s | ✅ 362 帧 / 15.083 s |
| 16.0 s | ❌ 最近的网格点 379 帧 = 15.79 s，超上限 |

帧数按 `17n + 5` **就近取整**（不是向上）：5.2 s 请求 125 帧 → 最近的有效值 124（5.17 s）；向上取整会跳到
141 帧（5.88 s）。界面会提示「请求 15.04 s，实际 15.08 s」这类差异。

实测：`--num-frames 360` 在 2 卡 CP 上生成 **362 帧 / 15.083 s**，耗时 85.4 s（640×384）。

## 分辨率：与 ComfyUI 一致

`webui/geometry.py` 复刻 ComfyUI `comfy_extras/nodes_resolution.py::ResolutionSelector.execute`：

```python
total_pixels = megapixels * 1024 * 1024          # 1 MP = 1024×1024，不是 1e6
scale = sqrt(total_pixels / (w_ratio * h_ratio))
width  = round(w_ratio * scale / multiple) * multiple
height = round(h_ratio * scale / multiple) * multiple
```

两个容易搞错的点：

- **`1 MP = 1024²`**。用 `1e6` 算 2 MP / 16:9 会得到 1904×1072，而 ComfyUI 给的是 **1920×1088**。
- **是四舍五入到倍数**，不是向下取整；`multiple=32` 时 2 MP / 16:9 → 1920×1088（1.99 MP）。

`tests/test_webui.py::test_canvas_matches_comfyui_at_multiple_8` 直接对着 ComfyUI 的公式做逐位比对，
6 个像素量 × 9 个宽高比全部相等，避免以后重构时悄悄漂移。

帧数按 `17n + 5` 网格**就近取整**：5.2 s 请求 125 帧，最近的有效值是 **124**（5.17 s）；如果向上取整会跳到
141 帧（5.88 s），那不是用户要的时长。

## 设备状态：物理容量 vs 可分配容量

Ascend 卡上这是两个不同的数，`webui/devices.py` 分开取、分开展示：

| 指标 | 来源 | 本机实测 |
|---|---|---|
| **可分配** | `torch.npu.mem_get_info()` | 61.27 GiB（进程能用的预算） |
| **物理 HBM** | `npu-smi info -t memory` | 64.0 GiB（板载总量） |
| 差值 | 相减 | ≈2.73 GiB（驱动 / CANN 预留） |

`npu-smi` 的 used 包含驱动常驻占用，所以**空闲时也有 2.8 GB**；`torch` 的 used 只统计本进程分配。
UI 两个都显示，不会拿一个冒充另一个。

为什么不直接解析 `npu-smi` 的表格：它是给人看的，列宽和列数随 CANN 版本、设备型号变化（本机的
`AICore(%)  Memory-Usage(MB)  HBM-Usage(MB)` 就挤在同一个单元格里，数字还右对齐，caption 位置和数字位置
对不上）。所以：

- **主来源是 torch API**（`mem_get_info` / `utilization` / `get_device_name`），稳定；
- `npu-smi` 只补 torch 拿不到的板卡信息（物理容量、温度、功率），按 caption 偏移解析，解析失败就是 `None`，
  绝不编造数字。

还有一个映射陷阱：`torch.npu` 的 device index 是 **chip logic id**，`npu-smi -i` 要的是 **物理 NPU id**。
本机 logic 0/1 都在物理 NPU 1 下，所以必须用 `npu-smi info -m` 建立映射再查，不能假设 `-i 0`。

## 默认就是最快配置

表单默认值由 `server.fastest_defaults()` 统一给出（`/api/config` 的 `defaults`），对应本机实测最快的组合：

| 项 | 默认 | 依据 |
|---|---|---|
| 设备数 | 全部可见 NPU（上限 4） | 2 卡 CP 比单卡快 1.65× |
| 并行 | 多卡 CP，单卡 single | 768p：CP 123 s vs TP 176 s vs 单卡 204 s |
| 分组卸载 | 开，**1 block/组** | 流式重叠只支持 1 block/组；4 block/组实测更慢（127 s vs 123 s） |
| 注意力后端 | `auto`（按序列长度选 kernel） | 见 `../NPU.md` 第 5 节 |
| 融合 RMSNorm | 开 | 同上 |

「运行时」标题旁的**最快配置**按钮会把设备数、并行、后端、卸载分组一键恢复成上表的值（不动提示词、画布、
时长、种子这些创作输入），并写进草稿，刷新后保持。

2 MP（1920×1088）同向：CP 452 s，单卡 844 s。

界面上的 `并行方式` 可以改成 TP 或单卡；选 TP 会自动关掉分组卸载（两者互斥）。

## 附件：首帧 / 尾帧 / 参考素材

三个模式下附件都用同一套「附件模型」：

- **首帧 / 尾帧**（关键帧模式）：两个独立槽位，缩略图直接可见；点空槽添加，点已填充的槽替换，右上角 `✕` 移除。
- **参考素材**（参考模式）：网格排布，缩略图左上角标 `P1/V1/A1…`，按排列顺序编号。`⟳` 替换单个条目，
  `✕` 删除单个条目，删除后编号自动重排。点「添加素材」可继续追加。
- 数量上限沿用后端校验（9 图 / 3 视频 / 3 音频，合计 12），超出时前端立即提示并拒绝多余的条目。
- 视频显示首帧 + `▶` 角标，音频显示 `♪`；图片直接用 `blob:` URL 预览。

文件不再从 `<input type=file>` 读取：输入框只作为选择器，选中的文件被复制进 `state.attachments`。这是删除
单个条目和替换单个条目能够实现的唯一方式（`FileList` 是只读的）。

## 刷新不丢输入

页面状态存在 `sessionStorage`（当前标签页），附件用 IndexedDB 镜像：

| 存什么 | 位置 | 说明 |
|---|---|---|
| 提示词、模式、宽高比、像素量/自定义尺寸、时长、种子、设备数、并行、注意力后端、各开关 | `sessionStorage['hyperflow.draft.v1']` | 输入时 250 ms 防抖写入 |
| 首帧 / 尾帧 / 参考素材的文件本体 | IndexedDB `hyperflow-media` / `files` | 每个附件一个 token，刷新后按 token 取回 |

因此刷新（甚至标签页崩溃后恢复）会带回完整草稿。用 `sessionStorage` 而不是 `localStorage` 是为了让新标签页
从默认值开始，而不是继承上一个标签页的半成品。

设备数如果用户没手动改过，仍由探测结果决定（换机器后自动适配）；手动改过则保留。

附件文件失效（比如浏览器清理了 IndexedDB）时会提示「N 个上次选择的文件已失效，请重新选择」，其余字段照常恢复。

## 完成后自动展示

2 秒一次的 `/api/status` 轮询会在任务**离开运行队列**时做一次定向查询（`/api/tasks/<id>`）：

- `done` → 直接把舞台切到该任务的视频、选中对应历史行、加载日志，并提示「生成完成」
- `failed` → 提示错误并打开它的日志
- `cancelled` / `interrupted` → 更新提示，不弹视频

提交后立刻记下 `pendingTaskId`，因此即使任务在第一次轮询之前就结束，也会被正确接住。以前只有 `refreshAll`
（刷新页面或点「刷新」）才会调用 `showVideo`，所以必须手动刷新才看得到结果。

历史列表也只在「有任务开始 / 结束」这两种跳变时重新拉取，不会每 2 秒刷一次。

## 舞台按视频真实比例

`.stage` 的 `aspect-ratio` 由 `showVideo()` 按任务的 `width/height` 设置，`video` 用 `object-fit: contain`：

| 画布 | 表现 |
|---|---|
| 16:9 / 21:9 等横屏 | 撑满中间栏宽度 |
| 9:16 / 1:1 等竖屏或方形 | 加 `.portrait`，宽度取 `min(100%, 56vh)`，避免把下方计划卡片挤出屏幕 |

`loadedmetadata` 还会用解码出的 `videoWidth/videoHeight` 再校正一次，作为任务记录与实际文件不一致时的兜底。
以前 `.stage` 锁死 `16 / 9`，任何非 16:9 的产物都会被放进 16:9 的框里，看起来像「比例不对」。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 页面 |
| GET | `/api/config` | 宽高比、默认值、限制、模型路径、设备 |
| GET | `/api/status` | 设备 + 队列（供 2 s 轮询） |
| GET | `/api/tasks` | 历史 |
| GET | `/api/tasks/<id>` | 单任务 |
| GET | `/api/tasks/<id>/log?lines=N` | 日志尾部（`text/plain`） |
| GET | `/api/tasks/<id>/video` | 产物 mp4 |
| POST | `/api/plan` | 只算画布，不运行 |
| POST | `/api/tasks` | 提交（multipart） |
| POST | `/api/tasks/<id>/cancel` | 取消（SIGTERM 整个进程组） |
| POST | `/api/tasks/<id>/retry` | 重跑 |

任务按 FIFO 串行执行（一张卡跑一个 2 MP 任务已经占满）。每个任务的提示词、上传、日志、产物都在
`webui/runs/<id>/`，服务重启后历史仍在，未完成的任务标为「已中断」。

## 测试

```bash
.venv312/bin/python -m pytest tests/test_webui.py -q      # 136 passed，约 6 s
```

覆盖：ComfyUI 分辨率逐位一致、帧网格、`npu-smi` 解析（温度/功率/用量，不含任何硬编码容量）、
物理 vs 可分离开、请求校验、CLI 参数拼装、HTTP 端点、路径穿越、附件模型与草稿持久化。

测试夹具里的容量是**任意取值**（40960 / 32768 MB），不是本机或任何特定卡的数字，换设备不会挂。
