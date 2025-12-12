import os
import pandas as pd

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(FILE_DIR, "..", "processed_data.pkl")
output_path = os.path.join(FILE_DIR, "..", "processed_normalized_data_V2.pkl")

data: pd.DataFrame = pd.read_pickle(data_path)
data.head()

# Do some basic filtering of the data
# For exampel paper https://iclr.cc/virtual/2025/poster/31514, has 450 coauthors, elading to a "kink" in the distribution
# Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models

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

all_authors = set()
for authors in data["authors"]:
    all_authors.update(authors)

print(f"Total unique authors: {len(all_authors)}")

papers_per_author = {}
for authors in data["authors"]:
    for author in authors:
        if author not in papers_per_author:
            papers_per_author[author] = 0
        papers_per_author[author] += 1

low_publish_authors = [
    author for author, count in papers_per_author.items() if count < 3
]
print(f"Authors with less than 3 papers: {len(low_publish_authors)}")

filtered_data = data.copy()
# # now remove the low publish authors from the list of authors in each paper
for i, authors in enumerate(filtered_data["authors"]):
    print(
        f"Removing low publish authors from paper {i+1}/{len(filtered_data)}", end="\r"
    )
    filtered_data.at[filtered_data.index[i], "authors"] = [
        author for author in authors if author not in low_publish_authors
    ]

# remove all papers that now have less than 2 authors
filtered_data = filtered_data[filtered_data["authors"].apply(len) >= 2]
# remove all papers that now have more than 20 authors
filtered_data = filtered_data[filtered_data["authors"].apply(len) <= 20]

filtered_data.reset_index(drop=True, inplace=True)
filtered_data.to_pickle(output_path)
