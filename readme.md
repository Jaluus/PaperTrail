# PaperTrail: Graph-Based Personalized Paper Recommendations for Conference Authors


## Abstract

We propose to build a personalized paper recommender for conference attendees using an author–paper bipartite graph.
Nodes represent authors and papers, and edges capture authorship relations.
We treat recommendations as ranking candidate papers for a target author at a given conference.
To incorporate semantic signals, we enrich the graph with textual embeddings derived from paper titles and abstracts, and systematically study their contribution to recommendation quality.
For evaluation, we employ a conference-based split in which the model is trained on one (or more) source conferences and evaluated on a disjoint target conference, enabling a realistic test of generalization performance.

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

## Folder Structure

```bash
data/  
    ├── json/                # Raw JSON files downloaded from the source  
    ├── raw_data.pkl         # Raw data in pickle format  
    ├── processed_data.pkl   # Processed data ready for graph construction
    ├── hetero_data_no_coauthor.pt # HeteroData object with degree filtering
    ├── hetero_data_filtered_3_2.pt # HeteroData object with degree filtering
    └── preparation_scripts/ # Scripts to prepare and process raw data
modeling/  
    ├── models/         # model class implementations (LightGCN, HeteroGCN, Text Dot Product Baseline)
scripts/
    └── download_data.sh     # Script to download the preprocessed data from AWS
```

## Graph Structure

**Node Types:** Authors, Papers  
**Edge Types:** Author writes Paper, Paper written by Author (reverse edges)
**Node Features:** Paper embeddings (Text embeddings from abstracts and titles), Author features (Vector of Ones)
**Dimensionality:** 256 for paper embeddings, generated using OpenAI's _text-embedding-3-large_ model.

<img src=figures/plot.png alt="Graph Structure" style="width:50%" />

## Metrics

Evaluating a recommender system can be tricky - even though we can view it as a classification or a link prediction
problem, simple metrics such as accuracy or AUC may not be able to capture the quality of recommendations.

Therefore, we evaluate our models using **personalized ranking metrics**: _Recall@K_ and _Precision@K_.
The metrics are computed per user in the test set and then averaged across all users.
- **Recall@K**: Measures the proportion of relevant items that are successfully retrieved in the top K recommendations.
  
  $$
  Recall@K = \frac{|\{relevant\_items\} \cap \{recommended\_items@K\}|}{|\{relevant\_items\}|}
  $$
- **Precision@K**: Measures the proportion of recommended items in the top K that are relevant.
  
  $$
  Precision@K = \frac{|\{relevant\_items\} \cap \{recommended\_items@K\}|}{K}
  $$

## Training the Recommender System Models

First, train the models:
* GraphSage: `python -m train`
* LightGCN: `python -m train --LightGCN`

You can run the `analyze_results.ipynb` notebook in the meantime to produce some plots of training metrics.

Afterwards, evaluate the models by using the `model_evaluation.ipynb` notebook.


## Loss Function

We utilize a Bayesian Personalized Ranking (BPR) loss, a pairwise objective which encourages the predictions of positive samples to be higher than negative samples for each user.

$$
L_{BPR} = - \frac{1}{|E_{pos}(u^*)|\cdot|E_{neg}(u^*)|} \sum_{(u^*,v_{pos}) \in E_{pos}(u^*)} \sum_{(u^*,v_{neg}) \in E_{neg}(u^*)} -log(f_\theta(u^*, v_{pos}) - f_\theta(u^*, v_{neg}))
$$

$\hat{y}_{u}$: predicted score of a positive sample

$\hat{y}_{uj}$: predicted score of a negative sample
