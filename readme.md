# PaperTrail: Graph-Based Personalized Paper Recommendations for Conference Authors


## Abstract

We introduce PaperTrail, a graph-based recommendation system designed to assist conference authors in discovering
interesting papers. We represent the relationships between authors and papers as a heterogeneous bipartite graph,
where nodes represent authors and papers, and edges denote authorship.
We attempt to use the graph structure as well as textual features (title and abstract) to recommend relevant papers to authors based
on their previous publications. We compare two graph neural network models, GraphSAGE and LightGCN,
against a non-graph baseline that simply uses text embeddings of the papers and dot product similarity.

See the associated blog post [here](https://medium.com/@jaluus/26c80a5a6a5a).

## Motivation

At the large conferences such as NeurIPS, ICML, and ICLR, the attendees can face information overload.
Conferences such as NeurIPS have started hosting over 5000 papers each year making it difficult to scan and figure out where to go based on your own interests.
Reading the abstracts of hundreds of papers is infeasible and time-consuming.

Recommender systems can help researchers plan their schedule, discover adjacent fields, propose interesting papers,
and foster collaboration between groups.  The goal of our system is to propose a ranked list of interesting papers
given an author one found interesting.

## Environment setup
Install the requirements:
```bash
pip install -r requirements.txt
```

Additionally, install PyTorch and PyG for your specific CUDA version by following the instructions at https://pytorch.org/get-started/locally/ and https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.

As an alternative, a pre-compiled environment packaged in a Docker container may be used. Using Singularity (Apptainer),
use the following command to start a shell within the container:
```bash
singularity shell -B / --nv docker://gkrz/lgatr:v3
```

## Dataset creation
The dataset may be obtained and preprocessed by running the following script:

```bash
sh scripts/collect_dataset.sh
```

However, to save time, we provide preprocessed data files:
- Raw JSON Data: [Download JSON Data](https://papertraildata.s3.us-west-1.amazonaws.com/json.zip)
- Raw Data: [Download Raw Data](https://papertraildata.s3.us-west-1.amazonaws.com/raw_data.pkl)
- Processed Data: [Download Processed Data](https://papertraildata.s3.us-west-1.amazonaws.com/processed_data.pkl)
- Processed Normalized Data: [Download Processed Data](https://papertraildata.s3.us-west-1.amazonaws.com/processed_normalized_data.pkl)
- Full Graph Data: [Download Full Graph Data](https://papertraildata.s3.us-west-1.amazonaws.com/hetero_data.pt)

The data may also be downloaded directly by running the script `scripts/download_data.sh`.

## Graph Structure

**Node Types:** Authors, Papers  
**Edge Types:** Author writes Paper, Paper written by Author (reverse edges)
**Node Features:** Paper embeddings (Text embeddings from abstracts and titles), Author features (Vector of Ones)
**Dimensionality:** 256 for paper embeddings, generated using OpenAI's _text-embedding-3-large_ model.

<img src=figures/plot.png alt="Graph Structure" style="width:50%" />


## Models

We implement and compare two graph-based recommender system models:
1. **GraphSage** [3]: A Graph Neural Network (GNN) that generates node embeddings by aggregating
features from a node's local neighborhood. We adapt GraphSage for our heterogeneous bipartite graph using PyG [2], see
`modeling/models/simpleGNN.py`.
2. **LightGCN** [4]: A simplified model using only neighborhood aggregation without feature transformation or 
nonlinear activation. It is transductive, as the embeddings are learned.
See `modeling/models/LightGCN.py` for the implementation.
3. **TextDotProduct**: A non-graph baseline that computes the dot product between author and paper text embeddings.
The author embeddings are computed as the mean of the embeddings of their authored papers that should be seen
during training. See `modeling/models/textDotProduct.py` for the implementation.
## Training
To reproduce the results using both models, run the following commands:

For GraphSAGE:
```bash
python -m train
```
For LightGCN: 
```bash
python -m train --LightGCN
```

To produce plots of training metrics, run the `analyze_results.ipynb` notebook.



### Loss Function

We utilize a Bayesian Personalized Ranking (BPR) loss [4], a pairwise objective which encourages the predictions of
positive samples to be higher than negative samples for each user.

See the [blog post](https://medium.com/@jaluus/26c80a5a6a5a) for more details.



## Evaluation

### Evaluation metrics

Evaluating a recommender system can be tricky - even though we can view it as a classification or a link prediction
problem, simple metrics such as accuracy or AUC may not be able to capture the quality of recommendations.

Therefore, we evaluate our models using **personalized ranking metrics**: _Recall@K_ and _Precision@K_.
The metrics are computed per user in the test set and then averaged across all users.
- **Recall@K**: Measures the proportion of relevant items that are successfully retrieved in the top K recommendations.
- **Precision@K**: Measures the proportion of recommended items in the top K that are relevant.

### Evaluating the trained models

Please refer to `model_evaluation.ipynb` for detailed evaluation of the trained models using the ranking metrics.

### Results

Our models achieve the following performance:

| Model           | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
|-----------------|--------:|---------:|---------:|---------:|-----------:|
| GraphSAGE       |  0.1572 |   0.2054 |   0.2582 |   0.3341 |     0.3962 |
| LightGCN        |  0.1760 |   0.2500 |   0.3359 |   0.4439 |     0.5132 |
| TextDotProduct  |  0.0506 |   0.0697 |   0.0944 |   0.1353 |     0.1768 |


## References

[1] He, Xiangnan, et al. “LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation” SIGIR 2020.

[2] Fey, Matthias, and Jan E. Lenssen. “Fast Graph Representation Learning with PyTorch Geometric” ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[3] Hamilton, William L., Rex Ying, and Jure Leskovec. “Inductive Representation Learning on Large Graphs” NeurIPS 2017.

[4] Rendle, Steffen, et al. “BPR: Bayesian Personalized Ranking from Implicit Feedback.” arXiv:1205.2618, arXiv, 9 May 2012.

