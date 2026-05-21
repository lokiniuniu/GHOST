# 👻 GHOST

**GHOST** is a causal visual geometry transformer that extends [VGGT](https://github.com/facebookresearch/vggt) with a training-free rolling KV-cache memory, enabling stable, infinite-horizon streaming 3D reconstruction from continuous image sequences. 🚀

---

## ✨ Key Features

- ♾️ **Infinite-horizon streaming inference** with bounded GPU memory via importance-based KV cache eviction
- 🧠 **Cosine-similarity-guided per-layer budget allocation**
- 🔄 **Alternating frame/global attention** for joint per-frame and cross-frame reasoning
- ⚡ Efficient long-sequence processing without OOM
- 📊 Evaluated on **7-Scenes** and **NRGBD** benchmarks
- 🏗️ Training-free streaming extension built on top of VGGT

---

# 🛠️ Environment Setup

**Requirements:** Python 3.11, CUDA 12.x, PyTorch 2.3.1

```bash
# 1️⃣ Clone the repository
git clone https://github.com/lokiniuniu/GHOST.git
cd GHOST

# 2️⃣ Create a conda environment
conda create -n ghost python=3.11 cmake=3.14.0
conda activate ghost

# 3️⃣ Install Python dependencies
pip install -r requirements.txt

# 4️⃣ (macOS / some Linux) Fix OpenMP conflict
conda install 'llvm-openmp<16'
```

> 💡 **Optional:** Install [`flash-kmeans`](https://github.com/cloneofsimo/flash-kmeans) for **2–5× faster** KV cache quantization encoding:
>
> ```bash
> pip install flash-kmeans
> ```

---

# 🤗 Model Weights

Download the pretrained **StreamVGGT** checkpoint from Hugging Face and place it under `./ckpt/`:

```bash
# Using huggingface-hub CLI
pip install -U huggingface_hub
huggingface-cli download lch01/StreamVGGT --local-dir ./ckpt
```

Or download manually from:

👉 https://huggingface.co/lch01/StreamVGGT

The checkpoint file should be at:

```text
./ckpt/model.pth
```

(or pass the path explicitly via `--checkpoint_path`)

---

# 📦 Dataset Download

## 🎬 7-Scenes

Download from the [official 7-Scenes page](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) and place under:

```text
./7scenes/
```

## 🌈 NRGBD

Download from [neural-rgbd-surface-reconstruction](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) and place under:

```text
./datasets/nrgbd/
```

---

# 🚀 Running Inference

```bash
# 🎥 Basic inference on a folder of images
python run_inference.py \
    --input_dir path/to/images/ \
    --checkpoint_path ./ckpt/model.pth

# ♾️ Long sequences:
# Stream frame-by-frame and write results to disk (avoids OOM)
python run_inference.py \
    --input_dir path/to/images/ \
    --checkpoint_path ./ckpt/model.pth \
    --frame_cache_dir path/to/output_per_frame/ \
    --no_cache_results
```

---

# 📈 Evaluation

## 🧪 7-Scenes / NRGBD

```bash
cd src

python -m accelerate.commands.launch eval/mv_recon/launch.py \
    --weights ../ckpt/model.pth \
    --model_name StreamVGGT \
    --dataset 7scenes \
    --scenes_root ../7scenes \
    --output_dir ../outputs/7scenes
```

## ⚙️ Key Evaluation Options

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `7scenes` | `7scenes`, `nrgbd`, or `all` |
| `--total_budget` | `1200000` | 🎯 Total KV token budget across all layers |
| `--use_cosine_budget` | `True` | 🧠 Cosine-similarity-based per-layer budget allocation |
| `--eviction_mode` | `importance` | 🗑️ KV eviction strategy |
| `--importance_weights_path` | *(default)* | ⚖️ Custom importance weights JSON |
| `--max_frames` | `None` | 🎞️ Limit frames per sequence |

---

# 🧱 Project Structure

```text
GHOST/
├── run_inference.py              # 🚀 Main inference entry point
├── requirements.txt              # 📦 Python dependencies
├── configs/
│   ├── importance_weights_default.json
│   │                               # ⚖️ Default importance eviction weights
│   └── kv_budget_proportions_cosine.json
│                                   # 🧠 Per-layer KV budget proportions
├── scripts/                        # 🛠️ Utility shell scripts
└── src/
    ├── add_ckpt_path.py            # 🔗 Checkpoint path helper
    ├── visual_util.py              # 🎨 Visualization utilities
    ├── streamvggt/
    │   ├── models/
    │   │   ├── streamvggt.py       # 👻 Main model
    │   │   └── aggregator.py       # 🔄 Frame/global attention aggregator
    │   ├── layers/
    │   │   ├── attention.py        # 🧠 KV cache + eviction attention
    │   │   ├── block.py            # 🧱 Transformer block
    │   │   ├── vision_transformer.py
    │   │   ├── rope.py             # 🌀 Rotary positional embedding
    │   │   ├── patch_embed.py
    │   │   ├── mlp.py
    │   │   └── swiglu_ffn.py
    │   ├── heads/
    │   │   ├── camera_head.py      # 📷 Camera estimation
    │   │   ├── dpt_head.py         # 🌍 Depth prediction
    │   │   ├── track_head.py       # 🎯 Tracking
    │   │   └── track_modules/
    │   ├── eviction/
    │   │   ├── importance_eviction.py
    │   │   │                        # 🗑️ Importance scoring
    │   │   └── importance_weights_from_hyperparams.py
    │   ├── quantization/           # ⚡ KV cache quantization
    │   ├── kernels/                # 🧩 CUDA/sparse attention kernels
    │   └── utils/
    ├── eval/
    │   └── mv_recon/
    │       ├── launch.py
    │       ├── data.py
    │       ├── compute_kv_budget_from_cosine_sim.py
    │       └── compute_kv_budget_strategies.py
    ├── dust3r/                     # 🏗️ DUSt3R geometry utilities
    ├── croco/                      # 🐊 CroCo backbone utilities
    └── vggt/                       # 📚 Original VGGT reference
```

---

# 🙏 Acknowledgements

This project builds upon the following excellent open-source works:

- [DUSt3R](https://github.com/naver/dust3r)
- [VGGT](https://github.com/facebookresearch/vggt)
- [StreamVGGT](https://github.com/wzzheng/StreamVGGT)
- [CUT3R](https://github.com/CUT3R/CUT3R)
- [Point3R](https://github.com/YkiWu/Point3R)
- [FastVGGT](https://github.com/mystorm16/FastVGGT)
- [TTT3R](https://github.com/Inception3D/TTT3R)

Huge thanks to the open-source community ❤️

---

# 📖 Citation

Citation information will be released soon.

```bibtex
@misc{ghost2026,
  title={GHOST},
  author={TBD},
  year={2026},
  note={Citation coming soon}
}
```

---

# 📜 License

See `LICENSE.txt`.

---

# 🌟 If You Find GHOST Useful...

Consider giving the repo a ⭐ on GitHub!
