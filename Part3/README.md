# Lab 2 Part 3：Calibration-free AdaptQuant 重現（Qwen3-4B）

本資料夾重現 **AdaptQuant**（per-tile MXFP4 / MXFP6 格式選擇，權重與 activation 都量化）。
量化 Qwen3-4B 時**完全不使用任何資料**：每個 16×32 tile 只依它自己的統計量
`stat = log2(amax / rms)` 和 14 個固定門檻（`thresholds.json`）決定格式。

| 方法（W + A 都量化） | WikiText-2 PPL | 權重平均 bits |
|---|---|---|
| FP16（未量化） | 13.64 | 16 |
| **AdaptQuant（本實作）** | **14.20** | 5.29（含 per-tile 格式表 5.37） |
| NVFP4（本實作，驗證量化器用） | 13.93 | 4.5 |
| 全部 MXFP4（本實作） | 18.64 | 4.25 |
| 全部 MXFP6（本實作） | 13.92 | 6.25 |
| 論文 AdaptQuant / NVFP4 / MXFP4（Qwen3-4B，RTX 5090，BF16） | 13.98 / 13.96 / 18.20 | 5.14 / 4.5 / 4.25 |

`bash run.sh` 已在全新的 Python 3.11 venv（只有 pip）中從頭驗證：輸出與上表相同，總時間 14 分 45 秒（含安裝套件）。

## 1. 環境
- Python 3.11、`torch==2.5.1+cu121`（與評分環境相同），單張 NVIDIA T4 16 GB。
- 其他套件版本固定於 `requirements.txt`（transformers 4.57.6、datasets 4.8.5、accelerate 1.15.0 …）。
- 不需要任何 Blackwell / 低精度 kernel：所有 MX 與 NVFP4 都是 **fake quantization**
  （量化後立即反量化回 fp16），在 T4 上即可執行。

## 2. 一鍵重現
```bash
bash run.sh
```
`run.sh` 會：
1. `pip install -r requirements.txt`（log 在 `logs/pip.log`）；
2. 用 `thresholds.json` 量化 Qwen3-4B 的全部 252 個 Linear（36 層 × q/k/v/o/gate/up/down）的權重，
   並在每次 forward 時即時量化它們的輸入 activation；
3. 以課程提供的 `evaluate.py` 中**未修改**的 `eval_ppl_wikitext2`（WikiText-2 test、seqlen 2048、fp16）計算 PPL。

預期輸出（stdout，約 10–15 分鐘，首次執行另需下載 Qwen3-4B 約 8 GB）：
```
mode: adaptquant
weight bits/element: 5.2943
ppl: 14.1993
```
詳細的格式比例在 `logs/adaptquant.log`（權重 tile：FP4 47.8%、FP6 52.2%、FP8 0%）。

## 3. 其他指令
```bash
# 自我檢查：元素格點、MX SQNR、FP4->FP6 無損擴寬、tile 選擇、門檻求解器（CPU 即可）
python test_mx.py

# 對照組（同樣是權重 + activation）
python run_ptq.py --mode nvfp4     # NVFP4：16 元素 block、E4M3 scale
python run_ptq.py --mode mxfp4     # 全部 MXFP4
python run_ptq.py --mode mxfp6     # 全部 MXFP6

# （選用）重新推導門檻：下載 Llama-3.1-8B（16 GB），需約 30 GB CPU RAM，約 10 分鐘
python derive.py --model unsloth/Llama-3.1-8B --out thresholds.json
```

## 4. 為什麼是 calibration-free
- **量化 Qwen3-4B 不需要任何資料**：`run_ptq.py` 在 PPL 評估之前沒有讓任何文字通過模型；
  權重格式由權重自己的 tile 統計量決定，activation 格式由每個 tile 進入 GEMM 時的統計量即時決定。
- **門檻是一次性、與目標模型無關的常數**：`thresholds.json` 的 14 個數字（2 種 operand × 7 種 projection）
  依論文 Algorithm 1 **只在參考模型 Llama-3.1-8B 上推導一次**（`derive.py`，用 WikiText-2 *train* 的前
  2048 個 token；test split 從未使用），條件是每一層的輸出加權 SQNR 不低於 NVFP4，再原封不動套用到 Qwen3-4B。
  這與論文的流程相同（論文同樣在 Llama-3.1-8B 上推導、套用到 13 個模型），沒有任何針對 Qwen3-4B 的調整。

## 5. 檔案
| 檔案 | 內容 |
|---|---|
| `mx.py` | MXFP4/6/8（E8M0 scale、k = 32）與 NVFP4 的 fake quantizer；tile 統計量與逐 tile 格式選擇 |
| `derive.py` | 門檻推導（Algorithm 1 + Appendix B：grid 0–4.5 step 0.02 + CAP，prefix sum 精確求解） |
| `thresholds.json` | 推導結果（每種 projection 的 t1、t2、bits、每層對 NVFP4 的最小 SQNR margin） |
| `run_ptq.py` | 量化 Qwen3-4B 並評估 PPL |
| `evaluate.py` | 課程提供的評估程式（未修改） |
| `test_mx.py` | 自我檢查 |
| `run.sh`、`requirements.txt` | 一鍵重現 |
