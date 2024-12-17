import os
import subprocess
import argparse
import json
import tempfile
from dotenv import load_dotenv


def load_class_mapping():
    json_path = "shapenet_classes.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Class mapping file '{json_path}' not found.")

    with open(json_path, 'r') as file:
        return json.load(file)


def download_and_extract_directly(class_name, class_mapping, huggingface_token, output_dir="../../ShapeNet_data"):
    class_code = class_mapping.get(class_name.lower())
    if not class_code:
        raise ValueError(f"Class '{class_name}' not found! Available classes: {list(class_mapping.keys())}")

    print(f"Found class code '{class_code}' for class name '{class_name}'.")

    url = f"https://huggingface.co/datasets/ShapeNet/ShapeNetCore/resolve/main/{class_code}.zip"
    class_folder = os.path.join(output_dir, class_name)

    os.makedirs(output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip") as temp_zip:
        print(f"Downloading ShapeNet class '{class_name}'...")
        wget_command = [
            "wget",
            "--header",
            f"Authorization: Bearer {huggingface_token}",
            "-q",
            "-O", temp_zip.name,
            url
        ]
        subprocess.run(wget_command, check=True)
        print(f"Downloaded zip file to temporary location: {temp_zip.name}")

        print(f"Extracting ShapeNet class '{class_name}'...")
        subprocess.run(["unzip", "-q", temp_zip.name, "-d", output_dir], check=True)

    extracted_folder = os.path.join(output_dir, class_code)
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, class_folder)
        print(f"Renamed folder to: {class_folder}")
    else:
        raise FileNotFoundError(f"Extracted folder '{extracted_folder}' not found.")

    print(f"Download and extraction for '{class_name}' completed successfully!\n")


if __name__ == "__main__":
    load_dotenv()
    huggingface_token = os.getenv("HF_TOKEN")

    if not huggingface_token:
        raise ValueError("HUGGINGFACE_TOKEN not found in .env file!")

    parser = argparse.ArgumentParser(description="Download and organize ShapeNet class data.")
    parser.add_argument(
        "class_names",
        type=str,
        nargs="+",
        help="Human-readable names of the classes to download (e.g., 'car airplane')."
    )

    args = parser.parse_args()

    class_mapping = load_class_mapping()

    for class_name in args.class_names:
        try:
            download_and_extract_directly(class_name, class_mapping, huggingface_token)
        except Exception as e:
            print(f"Error downloading class '{class_name}': {e}")
