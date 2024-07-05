import requests
import zipfile
import os


def download_dataset(
    dataset: str,
    data_path: str = "data",
    record_id: str = 12633988,
    redownload: bool = False,
):
    file_name = f"{dataset}.zip"
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(file_path) and not redownload:
        print(f"{dataset} already exists, skipping download.")
    else:
        # Zenodo API URL
        url = f"https://zenodo.org/api/records/{record_id}"

        # Fetch the record metadata
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Error fetching record: {response.status_code}")
        data = response.json()

        # Ensure the save path exists
        os.makedirs(data_path, exist_ok=True)

        # Download each file
        name_to_url = {file["key"]: file["links"]["self"] for file in data["files"]}
        file_url = name_to_url[file_name]
        # Download the file
        print(f"Downloading {dataset} from {file_url}...")
        response = requests.get(file_url)
        if response.status_code != 200:
            raise Exception(f"Error downloading file {dataset}: {response.status_code}")

        # Save the file
        with open(file_path, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(data_path)
        os.remove(file_path)  # Remove zip file after extraction

        print(f"CCLE data downloaded and extracted to {data_path}")
