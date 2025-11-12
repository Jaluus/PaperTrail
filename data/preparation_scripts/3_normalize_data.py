import os
import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(FILE_DIR, "..", "processed_data.pkl")
output_path = os.path.join(FILE_DIR, "..", "processed_normalized_data.pkl")

data: pd.DataFrame = pd.read_pickle(data_path)
data.head()

# Do some basic filtering of the data
# For exampel paper https://iclr.cc/virtual/2025/poster/31514, has 450 coauthors, elading to a "kink" in the distribution
# Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models

# remove all paper where the number of authors is greater than 50, these skew the data too much
data = data[data["authors"].apply(len) <= 50]
# remove all papers where the number of authors is 1, they dont provide enough edge information
data = data[data["authors"].apply(len) >= 2]

# lowercase all author names to avoid duplicates due to casing
data["authors"] = data["authors"].apply(
    lambda authors: [author.lower().strip() for author in authors]
)

# remove all author names that are empty strings or only whitespace
data["authors"] = data["authors"].apply(
    lambda authors: [author for author in authors if author.strip() != ""]
)

# Lowercase all the paper names as well
data["name"] = data["name"].apply(lambda name: name.lower().strip())

# remove any paper with an empty name
data = data[data["name"].apply(lambda name: name.strip() != "")]

# remove any abstracts that are empty strings or only whitespace
data = data[data["abstract"].apply(lambda abstract: abstract.strip() != "")]

# Remove any duplicate papers based on their name
data = data.drop_duplicates(subset=["name"])

data.reset_index(drop=True, inplace=True)
data.to_pickle(output_path)
