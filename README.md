# Edge AI Lab 2：LLM Compression

邊緣人工智慧課程 Lab 2 的實作，內容是對 Qwen3-4B 做量化、剪枝與低秩分解，以及重現 calibration-free 的 AdaptQuant。

| 部分 | 內容 | 結果 |
| --- | --- | --- |
| Part 1 | MobileNetV2 上的線性量化（Q1–Q10） | 所有 test 通過 |
| Part 2 | 壓縮 Qwen3-4B 的 Linear 層（四個 case） | 見下表 |
| Part 3 | 重現 AdaptQuant（per-tile MXFP4/MXFP6，不需校正資料） | WikiText-2 PPL 14.20 |

## Part 2 結果

| Case | 方法 | Compression Ratio | PPL |
| --- | --- | --- | --- |
| Base | FP16 | 0% | 13.64 |
| 1 | GPTQ INT4 g128 + gate_proj INT3 g128 | 75.58% | 14.02 |
| 2 | GPTQ INT3 g512 | 81.02% | 15.49 |
| 3 | Whitened SVD (MLP) + INT8 | 50.01%（SVD 20.88%） | 20.80 |
| 4 | 結構化剪枝 MLP channel + GPTQ INT3 g512 | 85.23%（剪枝 22.20%） | 18.33 |

PPL 為 WikiText-2 test、seqlen 2048、fp16。壓縮率依實際儲存 bits 計算，包含 scale、zero point 等額外參數。所有方法都是 post-training，沒有 fine-tuning。

## 環境

- Python 3.11、`torch==2.5.1+cu121`
- 一張 NVIDIA T4（16 GB）即可執行
- 其他套件版本見各資料夾的 `requirements.txt`

## 執行

```bash
# Part 2：安裝套件、壓縮四個 case、輸出壓縮率與 PPL（一張 T4 約 2.5 小時）
cd Part2
bash 315605019_run.sh

# Part 3：AdaptQuant 量化與評估（約 15 分鐘）
cd Part3
bash run.sh
```

第一次執行會從 Hugging Face 下載 Qwen3-4B（約 8 GB）。Part 3 的詳細說明見 [Part3/README.md](Part3/README.md)。

## 檔案

```
Part1/  315605019_Part1.ipynb          線性量化 notebook
Part2/  common.py                      GPTQ、MSE clipping、whitened SVD、bits 計算
        case1.py ~ case4.py            四個 case 的壓縮設定
        run_eval.py, evaluate.py       課程提供的評估程式（只加入壓縮呼叫與 compression_rate）
        315605019_run.sh               一鍵執行
        test_common.py                 壓縮率計算的單元測試
        315605019_report.pdf           報告
Part3/  mx.py                          MX / NVFP4 fake quantizer 與 per-tile 格式選擇
        derive.py                      在 Llama-3.1-8B 上推導門檻（Algorithm 1）
        run_ptq.py                     量化 Qwen3-4B 並評估
        thresholds.json                推導出的 14 個門檻
        315605019_report.pdf           報告
```

## 參考

- GPTQ: Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023
- SVD-LLM: Wang et al., *SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression*, ICLR 2025
- AdaptQuant: *Hardware-Aligned Per-Tile MX Format Selection for Post-Training Quantization on NVIDIA Blackwell*（ICLR 2027 審稿中）
