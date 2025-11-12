import os
import json
import pandas as pd

#   {
#     "type": "Poster",
#     "name": "ABCD: Arbitrary Bitwise Coefficient for De-Quantization",
#     "virtualsite_url": "https://cvpr.thecvf.com//virtual/2023/poster/22138",
#     "speakers/authors": "Woo Kyoung Han, Byeonghun Lee, Sang Hyun Park, Kyong Hwan Jin",
#     "abstract": "Modern displays and contents support more than 8bits image and video. However, bit-starving situations such as compression codecs make low bit-depth (LBD) images (<8bits), occurring banding and blurry artifacts. Previous bit depth expansion (BDE) methods still produce unsatisfactory high bit-depth (HBD) images. To this end, we propose an implicit neural function with a bit query to recover de-quantized images from arbitrarily quantized inputs. We develop a phasor estimator to exploit the information of the nearest pixels. Our method shows superior performance against prior BDE methods on natural and animation images. We also demonstrate our model on YouTube UGC datasets for de-banding. Our source code is available at https://github.com/WooKyoungHan/ABCD"
#   },

file_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(file_dir, "..", "json")
output_file = os.path.join(file_dir, "..", "raw_data.pkl")


file_names = [f for f in os.listdir(data_dir) if f.endswith(".json")]

df = pd.DataFrame(
    {
        "name": pd.Series(dtype="string"),
        "abstract": pd.Series(dtype="string"),
        "url": pd.Series(dtype="string"),
        "authors": pd.Series(dtype="object"),
        "conference": pd.Series(dtype="string"),
        "year": pd.Series(dtype="Int64"),  # pandas nullable integer dtype
        "embedding": pd.Series(dtype="object"),  # store lists/ndarrays
    }
)

for file_name in file_names:
    file_path = os.path.join(data_dir, file_name)

    _, conference, year = file_name.rstrip(".json").split("_")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {file_name} with {len(data)} entries.")

    processed_entries = []
    for entry in data:
        name = entry.get("name", "")
        abstract = entry.get("abstract", "")
        url = entry.get("virtualsite_url", "")
        authors = entry.get("speakers/authors", "").strip().split(", ")
        if not abstract:
            print(f"Skipping entry with empty abstract: {name}")
            continue

        processed_entries.append(
            {
                "name": name,
                "abstract": abstract,
                "url": url,
                "authors": authors,
                "conference": conference,
                "year": int(year),
                "embedding": None,  # Placeholder for embedding
            }
        )

    df = pd.concat([df, pd.DataFrame(processed_entries)], ignore_index=True)

df.to_pickle(output_file)
