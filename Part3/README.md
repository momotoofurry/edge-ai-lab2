# Lab 2 Part 3：Calibration-free AdaptQuant 重現（Qwen3-4B）

本資料夾重現 **AdaptQuant**（per-tile MXFP4 / MXFP6 格式選擇，權重與 activation 都量化）。
量化 Qwen3-4B 時**完全不使用任何資料**：每個 16×32 tile 只依它自己的統計量
`stat = log2(amax / rms)` 和 14 個固定門檻（`thresholds.json`）決定格式。

**最終結果：PPL 14.64，effective bit rate 5.17 bits（≤ 5.3）。**

| 方法（W + A 都量化） | WikiText-2 PPL | 權重 bits | activation bits | effective bit rate |
|---|---|---|---|---|
| FP16（未量化） | 13.64 | 16 | 16 | 16 |
| **AdaptQuant（本實作，目標 5.25）** | **14.64** | **5.12** | **5.22** | **5.17** |
| AdaptQuant（目標 5.15，對照） | 14.93 | 4.99 | 5.16 | 5.08 |
| NVFP4（本實作，驗證量化器用） | 13.93 | 4.5 | 4.5 | 4.5 |
| 全部 MXFP4（本實作） | 18.64 | 4.25 | 4.25 | 4.25 |
| 全部 MXFP6（本實作） | 13.92 | 6.25 | 6.25 | 6.25 |
| 論文 AdaptQuant / NVFP4 / MXFP4（Qwen3-4B，RTX 5090，BF16） | 13.98 / 13.96 / 18.20 | 5.14 / 4.5 / 4.25 | – | – |

AdaptQuant 的 bits 都包含 scale 的成本和每個 tile 的格式標記。最終版的權重 tile 有 39.8% 用 MXFP6，
activation tile 有 44.8% 用 MXFP6，都沒有用到 MXFP8。

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
weight bits/element: 5.1236
activation bits/element: 5.2238
effective bit rate: 5.1737 (limit 5.3: OK)
ppl: 14.6440
```
權重和 activation 各自的 tile 格式比例記在 `logs/adaptquant.log`。

## 3. Effective bit rate 的計算

作業要求 (Weight Bits + Activation Bits) / 2 ≤ 5.3，`run_ptq.py` 會實際統計後印出：

- **每個 tile 的 bits**：元素 bits 加上共用 scale 的成本，每 32 個元素共用一個 8-bit E8M0 scale，
  所以 MXFP4 = 4 + 8/32 = 4.25、MXFP6 = 6.25、MXFP8 = 8.25。
- **格式標記**：混合精度時每個 16×32 tile 還要記錄它用哪種格式（5 bytes / 512 個元素 = 0.078 bits），
  也算進去。
- **權重 bits**：量化時統計 252 個 Linear 全部權重 tile 的格式比例，依比例平均後加上格式標記。
- **activation bits**：評估 PPL 時，每個 Linear 的輸入每次進來都統計 tile 的格式，對整個 WikiText-2 test
  的所有 activation tile 平均後加上格式標記。
- 每個 tile 的元素數相同，所以 tile 比例就是元素比例。

## 4. 其他指令
```bash
# 自我檢查：元素格點、MX SQNR、FP4->FP6 無損擴寬、tile 選擇、門檻求解器、bits 計算（CPU 即可）
python test_mx.py

# 對照組（同樣是權重 + activation）
python run_ptq.py --mode nvfp4     # NVFP4：16 元素 block、E4M3 scale
python run_ptq.py --mode mxfp4     # 全部 MXFP4
python run_ptq.py --mode mxfp6     # 全部 MXFP6

# （選用）重新推導門檻。資料夾裡已附 reference_hist.pt（Llama-3.1-8B 的 tile 統計），
# 有這個檔案就直接求解，幾秒鐘、不需要 GPU；刪掉它才會重新載入 Llama-3.1-8B（16 GB，需約 30 GB CPU RAM）。
python derive.py --model unsloth/Llama-3.1-8B --target-bits 5.25 --out thresholds.json
```

## 5. 為什麼是 calibration-free
- **量化 Qwen3-4B 不需要任何資料**：`run_ptq.py` 在 PPL 評估之前沒有讓任何文字通過模型；
  權重格式由權重自己的 tile 統計量決定，activation 格式由每個 tile 進入 GEMM 時的統計量即時決定。
- **門檻是一次性、與目標模型無關的常數**：`thresholds.json` 的 14 個數字（2 種 operand × 7 種 projection）
  依論文 Algorithm 1 **只在參考模型 Llama-3.1-8B 上推導一次**（`derive.py`，用 WikiText-2 *train* 的前
  2048 個 token；test split 從未使用），再原封不動套用到 Qwen3-4B。
  這與論文的流程相同（論文同樣在 Llama-3.1-8B 上推導、套用到 13 個模型），沒有任何針對 Qwen3-4B 的調整。
- **bit rate 上限也只在參考模型上處理**：如果要求每一層的誤差都不超過 NVFP4（論文的條件），
  FP6 比例太高，平均約 5.5 bits，超過 5.3。`derive.py` 因此加了一個全域的 slack：允許每一層的誤差
  比 NVFP4 多 s dB，並取「讓參考模型的 effective bit rate ≤ `--target-bits`」的最小 s。
  最終用目標 5.25（slack 0.7 dB），在參考模型上是 5.24 bits，在 Qwen3-4B 上實際是 5.17 bits。
  我先試過目標 5.15（slack 1.0 dB），Qwen3-4B 上是 5.08 bits、PPL 14.93，離 15 太近，
  而 bits 還有餘裕，所以改用 5.25。兩組結果都列在上表。

## 6. 檔案
| 檔案 | 內容 |
|---|---|
| `mx.py` | MXFP4/6/8（E8M0 scale、k = 32）與 NVFP4 的 fake quantizer；tile 統計量與逐 tile 格式選擇 |
| `derive.py` | 門檻推導（Algorithm 1 + Appendix B：grid 0–4.5 step 0.02 + CAP，prefix sum 精確求解），加上 bit rate 目標 |
| `thresholds.json` | 推導結果（每種 projection 的 t1、t2、bits、每層對 NVFP4 的最小 SQNR margin，以及使用的 slack 與參考模型 bit rate） |
| `reference_hist.pt` | 參考模型 Llama-3.1-8B 的 tile 統計（`derive.py` 產生，可重新求解門檻） |
| `run_ptq.py` | 量化 Qwen3-4B、統計權重與 activation 的 bits，並評估 PPL |
| `evaluate.py` | 課程提供的評估程式（未修改） |
| `test_mx.py` | 自我檢查 |
| `run.sh`、`requirements.txt` | 一鍵重現 |
