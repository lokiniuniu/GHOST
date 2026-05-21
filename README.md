# 👻 GHOST

**GHOST** is a causal visual geometry transformer that extends [VGGT](https://github.com/facebookresearch/vggt) with a training-free rolling KV-cache memory, enabling stable, infinite-horizon streaming 3D reconstruction from continuous image sequences. 🚀

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
