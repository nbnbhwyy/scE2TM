# $scE^2TM$: Toward Interpretable Single-Cell Embedding via Topic Modeling

[![PyPI version](https://badge.fury.io/py/scE2TM.svg)](https://badge.fury.io/py/scE2TM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The full description of $scE^2TM$ and its application on published single-cell RNA-seq datasets are available in our paper.

## 📖 Overview

$scE^2TM$ is a novel framework for interpretable single-cell embedding learning via topic modeling. It integrates multi-modal information and learns sparse topic-gene dependencies for improved interpretability.

### 1. Schematic overview of $scE^2TM$

![](Flow.jpg)

**(a)** To better collaborate the information of different modalities, clusters and topic heads are trained based on mutually refined neighborhood information by encouraging consistent clustering assignments of mutual nearest neighbors of the corresponding cells of different modalities in the embedding space.

**(b)** ECR clusters gene embeddings $g_j$ (•) as samples and topic embeddings $t_k$ (★) as centers with soft assignment $\pi^{*}_{\epsilon,jk}$. Here, ECR pushes $g_1$ and $g_2$ close to $t_1$, and away from $t_3$ and $t_5$.

**(c)** Sparse linear decoders learn topic embeddings and gene embeddings as well as sparse topic-gene dependencies during reconstruction, thus ensuring model interpretability.

## 🔧 Installation

> **Note**: The complete installation process (including environment setup and dependency installation) takes approximately **1 hour**.

### 1. Create a conda environment

```bash
conda create --name scE2TM_env python=3.8.8 -y
conda activate scE2TM_env
```

### 2. Install PyTorch

```bash
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install $scE^2TM$

You can install $scE^2TM$ using either of the following methods:

#### Option A: Install from PyPI (Recommended for users)

```bash
pip install scE2TM
```

This will automatically install all required dependencies.

#### Option B: Install from source (Recommended for developers)

```bash
# Clone the repository
git clone https://github.com/nbnbhwyy/scE2TM.git
cd scE2TM

# Install dependencies
pip install -r requirements.txt
```
<<<<<<< HEAD

## 🚀 Quick Start
=======
Installation typically completes in approximately 1.5 hours.
## 3 Usage
>>>>>>> 23659e43373cf98fd9a55cf07c85e5a7aabe158e

### Data Format

$scE^2TM$ requires the following input files in CSV format:
- **Gene expression matrix**: cell-by-gene matrix (`*_HIGHPRE.csv`)
- **Cell type annotations**: ground truth labels (`*_cell_anno.csv`) *used only for evaluation*
- **Foundation model embeddings**: pre-trained cell embeddings (`*.csv`)

We provide the **Wang** dataset as a default example for users to understand and debug the code.

### Running $scE^2TM$

#### Method 1: Using the source code

After installing from source, you can run the model with various options:

**Basic run with default parameters:**
```bash
python run.py
```
<<<<<<< HEAD
=======
On the provided example dataset, the demo completes in about one minute.

### Tutorial
>>>>>>> 23659e43373cf98fd9a55cf07c85e5a7aabe158e

**Specify dataset and number of topics:**
```bash
python run.py --dataset_name Wang --num_topics 50
```

**Choose GPU device (use -1 for CPU):**
```bash
# Run on GPU 0 (default)
python run.py --gpu_id 0

# Run on CPU
python run.py --gpu_id -1
```

**Full parameter example:**
```bash
python run.py \
    --dataset_name Wang \
    --num_topics 100 \
    --num_neighbors 15 \
    --num_top_genes 10 \
    --tac_weight 1.0 \
    --gpu_id 0 \
    --data_dir ./data \
    --output_dir ./output
```

#### Method 2: Using the PyPI package

We provide an interactive tutorial notebook that demonstrates the usage:

```bash
jupyter notebook scE2TM_demo_on_Wang_dataset.ipynb
```

### Output Files

After successful execution, the following files will be saved in the `output/Wang/` directory:

| File | Description |
|------|-------------|
| `Wang.pth` | Trained model checkpoint |
| `Wang_topic_distribution.csv` | Cell-topic distribution (theta) |
| `Wang_topic_embedding.csv` | Topic embeddings |
| `Wang_gene_embedding.csv` | Gene embeddings |
| `Wang_tg.csv` | Topic-gene distribution (beta) |

## 📚 Tutorials

We provide three comprehensive tutorials in the `tutorial/` directory that introduce the usage of $scE^2TM$ and reproduce the main quantitative results:

| Tutorial | Description |
|----------|-------------|
| [Clustering and Interpretable Evaluation](tutorial/Clustering-and-Interpretable-Evaluation.ipynb) | Evaluate clustering performance and interpret topic-gene relationships |
| [Pathway Enrichment](tutorial/Pathway-Enrichment.ipynb) | Enrichment analysis on learned topics to identify biological pathways |
| [Topic Gene Embedding](tutorial/Topic-gene-embedding.ipynb) | Visualize and analyze topic-gene embeddings |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions and support, please open an issue on GitHub or contact: [13247702278@163.com](mailto:13247702278@163.com)