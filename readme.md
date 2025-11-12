# PaperTrail: Graph-Based Personalized Paper Recommendations for Conference Authors

## Abstract

We propose to build a personalized paper recommender for conference attendees using an author–paper bipartite graph.
Nodes represent authors and papers, and edges capture authorship relations.
We treat recommendations as ranking candidate papers for a target author at a given conference.
To incorporate semantic signals, we enrich the graph with textual embeddings derived from paper titles and abstracts, and systematically study their contribution to recommendation quality.
For evaluation, we employ a conference-based split in which the model is trained on one (or more) source conferences and evaluated on a disjoint target conference, enabling a realistic test of generalization performance.

## Problem Statement

At the large conferences such as NeurIPS, ICML, and ICLR, the attendees can face information overload.
Conferences such as NeurIPS have started hosting over 5000 papers each year making it difficult to scan and figure out where to go based on your own interests.
Reading the abstracts of hundreds of papers is infeasible and time consuming.

Recommender systems can help researchers plan their schedule, discover adjacent fields, propose interesting papers, and foster collaboration between groups.
The goal of our system is to propose a ranked list of interesting papers given an author one found interesting.

## Data Downloads

- Raw JSON Data: [Download JSON Data](https://papertraildata.s3.us-west-1.amazonaws.com/json.zip)
- Raw Data: [Download Raw Data](https://papertraildata.s3.us-west-1.amazonaws.com/raw_data.pkl)
- Processed Data: [Download Processed Data](https://papertraildata.s3.us-west-1.amazonaws.com/processed_data.pkl)
- Full Graph Data: [Download Full Graph Data](https://papertraildata.s3.us-west-1.amazonaws.com/hetero_data.pt)
- Graph Without Coauthors Data: [Download Graph Without Coauthors Data](https://papertraildata.s3.us-west-1.amazonaws.com/hetero_data_no_coauthor.pt)
- Early models: https://huggingface.co/gregorkrzmanc/papertrail_models/tree/main

## Folder Structure

```bash
data/  
    ├── json/                # Raw JSON files downloaded from the source  
    ├── raw_data.pkl         # Raw data in pickle format  
    └── processed_data.pkl   # Processed data ready for graph construction  
```

## Graph Structure

**Node Types:** Authors, Papers  
**Edge Types:** Author-Paper, Author-Author (co-authorship)  
**Node Features:** Paper embeddings (Text embeddings from abstracts), Author features (Vector of Ones)

**Dimensionality:** 256 for paper embeddings
