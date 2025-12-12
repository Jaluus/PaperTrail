# PaperTrail: Graph-Based Personalized Paper Recommendations for Conference Authors

<div align="center">

A graph-based recommendation system for helping conference authors discover relevant papers using an author–paper graph and text embeddings. This project has been created for [Stanford CS224W: Graph Neural Networks](https://web.stanford.edu/class/cs224w/).

[[Blog post]](https://medium.com/@jaluus/26c80a5a6a5a) · [[Quickstart]](#-quickstart) · [[Data]](#-data) · [[Training]](#-training) · [[Evaluation]](#-evaluation)

</div>

<p align="center">
  <img src="figures/plot.png" alt="PaperTrail heterogeneous graph (authors ↔ papers)" width="400" />
</p>
<p align="center">
  <em>Figure 1: PaperTrail represents authors and papers as a bipartite graph; an edge indicates authorship. While the author features are initialized as vectors of 1s, the node features of the papers are initialized by their text embeddings from the abstracts and titles.</em>
</p>

## Abstract

PaperTrail represents relationships between authors and papers as a heterogeneous bipartite graph: nodes are authors and papers, and edges denote authorship.
We combine this graph structure with textual features (title + abstract embeddings) to recommend papers to authors based on their past publications.
We compare two graph-based recommenders (GraphSAGE and LightGCN) against a non-graph baseline that uses text embedding similarity.

## Motivation

Large conferences (e.g., NeurIPS, ICML, ICLR) publish thousands of papers per year, making it hard to quickly find relevant work.
PaperTrail is a small research project exploring whether an author–paper graph can improve personalized paper recommendations beyond text-only similarity.

## Features

- **Graph recommenders**: GraphSAGE (`modeling/models/simpleGNN.py`) and LightGCN (`modeling/models/lightGCN.py`).
- **Text-only baseline**: dot-product similarity (`modeling/models/TextDotProduct.py`).
- **Ranking evaluation**: Recall@K and Precision@K.
- **Preprocessed graph**: ready-to-train PyG `HeteroData` (`data/hetero_data.pt`).

## Quickstart

0. Install Python, this repo has been tested with Python 3.10.

1. Install base requirements:

```bash
pip install -r requirements.txt
```

2. Install PyTorch + PyTorch Geometric for your CUDA/CPU setup:

- PyTorch: <https://pytorch.org/get-started/locally/>
- PyG: <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>

3. Download the preprocessed graph data:

```bash
mkdir -p data
bash scripts/download_data.sh
```

4. Train a model:

```bash
# GraphSAGE (default)
python train.py

# LightGCN
python train.py --LightGCN
```

Outputs:

- Model checkpoints: `checkpoints/model_*`
- Training curves / metrics: `out/loss_*.pkl`, `out/metrics_*.pkl`

## Installation notes

### Container (optional)

A pre-built environment is available via Docker, and can be run using Singularity (Apptainer):

```bash
singularity shell -B / --nv docker://gkrz/lgatr:v3
```

## Data

Training expects a PyG `HeteroData` graph at `data/hetero_data.pt`

### Preprocessed downloads

| Artifact                         | Link                                                                                                             |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Raw JSON Data                    | [json.zip](https://papertraildata.s3.us-west-1.amazonaws.com/json.zip)                                           |
| Raw Data                         | [raw_data.pkl](https://papertraildata.s3.us-west-1.amazonaws.com/raw_data.pkl)                                   |
| Processed Data (with embeddings) | [processed_data.pkl](https://papertraildata.s3.us-west-1.amazonaws.com/processed_data.pkl)                       |
| Processed Normalized Data        | [processed_normalized_data.pkl](https://papertraildata.s3.us-west-1.amazonaws.com/processed_normalized_data.pkl) |
| Full Graph Data (PyG)            | [hetero_data.pt](https://papertraildata.s3.us-west-1.amazonaws.com/hetero_data.pt)                               |

Or fetch them into `./data`:

```bash
mkdir -p data
bash scripts/download_data.sh
```

### (Optional) Build the dataset yourself

You can try running the end-to-end preprocessing pipeline:

```bash
bash scripts/collect_dataset.sh
```

Notes:

- The scraping step may break if conference sites change.
- Embeddings are computed via OpenAI in `data/preparation_scripts/2_add_embeddings_to_df.py` (you must set `API_KEY` there).
- The preprocessing scripts currently write `data/hetero_data_V2.pt`; to train with `train.py`, either rename it or update `DATA_PATH` in `train.py`.

## Models

1. **GraphSAGE** [1] (heterogeneous variant via PyG [2]): `modeling/models/simpleGNN.py`
2. **LightGCN** [3] (transductive embeddings): `modeling/models/lightGCN.py`
3. **TextDotProduct** (text-only baseline): `modeling/models/TextDotProduct.py`

### Loss function

We train with **Bayesian Personalized Ranking (BPR)** [4], a pairwise objective that encourages positive edges to score higher than negatives.

$$\text{Loss}(u^\*) = \frac{1}{|E(u^\*)|\;|E_{\text{neg}}(u^\*)|} \sum_{(u^\*, v_{\text{pos}})\in E(u^\*)} \sum_{(u^\*, v_{\text{neg}})\in E_{\text{neg}}(u^\*)} -\log\!\left( \sigma\big(f_\theta(u^\*, v_{\text{pos}}) - f_\theta(u^\*, v_{\text{neg}})\big) \right)$$

Here $\sigma$ is the sigmoid function, and $f_\theta$ is the model's score function, which is typically a dot product of node embeddings.
$u^\*$ denotes the current author, and $E(u^\*)$ and $E_{\text{neg}}(u^\*)$ are the sets of positive and negative edges for that author.

This loss is averaged over some subset $U$ of sampled authors to form the mini-batch loss.

$$\text{Loss} = \frac{1}{|U|}\sum_{u^\*\in U} \text{Loss}(u^\*)$$

## Training

```bash
# GraphSAGE
python train.py

# LightGCN
python train.py --LightGCN
```

To visualize training curves, see `analyze_results.ipynb`.

## Evaluation

### Metrics

We evaluate using personalized ranking metrics computed per author and averaged:

- **Recall@K**: fraction of relevant papers retrieved in the top K.
- **Precision@K**: fraction of top K recommendations that are relevant.

For a detailed evaluation workflow, see `model_evaluation.ipynb`.

## Results

| Model          | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
| -------------- | -------: | --------: | --------: | --------: | ---------: |
| LightGCN       |   0.1760 |    0.2500 |    0.3359 |    0.4439 |     0.5132 |
| GraphSAGE      |   0.1572 |    0.2054 |    0.2582 |    0.3341 |     0.3962 |
| TextDotProduct |   0.0506 |    0.0697 |    0.0944 |    0.1353 |     0.1768 |

## Repository structure

```bash
PaperTrail/
├── train.py                          # training entrypoint
├── modeling/                         # model code + training utilities
│   ├── models/                       # recommender model implementations
│   │   ├── simpleGNN.py              # GraphSAGE-based heterogeneous GNN
│   │   ├── lightGCN.py               # LightGCN (learned embeddings + message passing)
│   │   └── TextDotProduct.py         # text-only baseline (dot product of text embeddings)
│   ├── metrics.py                    # Recall@K / Precision@K evaluation
│   ├── losses.py                     # BPR loss
│   └── sampling.py                   # minibatch (positive/negative) sampling
├── data/
│   ├── hetero_data.pt                # preprocessed PyG `HeteroData` used by `train.py`
│   └── preparation_scripts/          # raw-data → embeddings → graph preprocessing pipeline
│       ├── 0_download_data.py        # scrape/download conference JSON data
│       ├── 1_json_to_df.py           # convert JSON dumps to a pandas DataFrame
│       ├── 2_add_embeddings_to_df.py # add OpenAI text embeddings (requires API key)
│       ├── 3_normalize_data.py       # clean/filter/normalize author + paper fields
│       ├── 4_pandas_to_pyg.py        # build a PyG `HeteroData` author–paper graph
│       └── 5_filter_data.py          # optional degree-based graph filtering
├── scripts/
│   ├── download_data.sh              # download preprocessed artifacts into `data/`
│   └── collect_dataset.sh            # run the full preprocessing pipeline
├── checkpoints/                      # saved model weights
├── out/                              # training metrics/loss pickles
├── figures/
│   └── plot.png                      # graph visualization used in this README
├── analyze_results.ipynb             # plots training curves from `out/`
├── model_evaluation.ipynb            # evaluates trained models / computes ranking metrics
├── requirements.txt                  # base Python requirements (excluding torch/pyg)
└── readme.md                         # project overview and usage
```

## References

[1] Hamilton, William L., Rex Ying, and Jure Leskovec. “Inductive Representation Learning on Large Graphs” NeurIPS 2017.

[2] Fey, Matthias, and Jan E. Lenssen. “Fast Graph Representation Learning with PyTorch Geometric” ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[3] He, Xiangnan, et al. “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation” SIGIR 2020.

[4] Rendle, Steffen, et al. “BPR: Bayesian Personalized Ranking from Implicit Feedback.” arXiv:1205.2618, arXiv, 9 May 2012.

## License

This repository does not currently include a `LICENSE` file. If you plan to use this code beyond personal/research inspection, clarify licensing first.
