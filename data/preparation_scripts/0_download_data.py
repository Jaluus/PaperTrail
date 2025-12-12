import os

import bs4
import requests

urls = {
    "neurips": "https://neurips.cc/Downloads",
    "iclr": "https://iclr.cc/Downloads",
    "icml": "https://icml.cc/Downloads",
    "iccv": "https://iccv.thecvf.com/Downloads",
    "eccv": "https://eccv.ecva.net/Downloads",
    "cvpr": "https://cvpr.thecvf.com/Downloads",
}

file_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(file_dir, "..", "json")

for name, url in urls.items():

    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")
    contribs = soup.find_all("a", class_="conference")

    for contrib in contribs:
        year = int(contrib.text.strip())
        download_url = f"{url}/{year}"

        print(f"Downloading from Conference: {name} / {year}")

        response = requests.post(
            download_url,
            headers={
                "Referer": download_url,
            },
            data={
                "csrfmiddlewaretoken": "7dEoJiHJcTjL5zylb1ZPoKK90NS9fUqBcXmTkGeXcwASScwczM3F05ExOA8ZS3au",
                "file_format": "5",
                "posters": "on",
                "submitaction": "Download Data",
            },
            cookies={
                "browser_timezone": "America/Los_Angeles",
                "csrftoken": "fUSFLyHoaNrhXN81yVe0Mv4yYXq0NjU3",
                "sessionid": "bzohhqv37wj20r5pipnipjmgxyp54yj2",
            },
        )

        save_path = os.path.join(save_dir, f"data_{name}_{year}.json")
        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded data for {name} {year}")
