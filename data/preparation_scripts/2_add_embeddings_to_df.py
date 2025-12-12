import os

import openai
import pandas as pd

API_KEY = "your-openai-api-key"
client = openai.OpenAI(api_key=API_KEY)

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(FILE_DIR, "..", "raw_data.pkl")
output_file = os.path.join(FILE_DIR, "..", "processed_data.pkl")

df = pd.read_pickle(data_file)

abstracts = df["abstract"].tolist()
titles = df["name"].tolist()
prompts = [
    f"Title: {title}\n\nAbstract: {abstract}"
    for title, abstract in zip(titles, abstracts)
]


for i in range(0, len(prompts), 512):
    print(f"Processing batch {i // 512 + 1} / {(len(prompts) - 1) // 512 + 1}")
    batch = prompts[i : i + 512]
    response = client.embeddings.create(
        input=batch,
        model="text-embedding-3-large",
        dimensions=256,
    )
    embeddings = [response.data[j].embedding for j in range(len(response.data))]
    for j, embedding in enumerate(embeddings):
        df.at[i + j, "embedding"] = embedding

df.to_pickle(output_file)
