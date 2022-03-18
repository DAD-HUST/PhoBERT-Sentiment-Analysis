import pandas as pd
import os
import traceback
import time
import logging
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

parser = argparse.ArgumentParser(description="Create data from csv files")
parser.add_argument("--data_path", type=str, default="data/Sentiment/")
parser.add_argument("--output_path", type=str, default="data/")
parser.add_argument("--output_file", type=str, default="train.csv")

args = parser.parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)


error_file = []
data = dict(text=[],label=[])

logging.info("Started reading data from {}".format(args.data_path))
for folder in os.listdir(args.data_path):
    try:
        folder_path = os.path.join(args.data_path, folder)
        logging.info("Reading folder {} with {} files".format(folder_path, len(os.listdir(folder_path))))
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if file.endswith(".csv"):
                    df = pd.read_csv(
                        file_path,
                        index_col=False, 
                        names=["text"], 
                        encoding="utf-8",
                    )
                    df["label"] = folder
                    data["text"].extend(df["text"].tolist())
                    data["label"].extend(df["label"].tolist())

                    del df
                elif file.endswith(".txt"):
                    with open(file_path, 'r') as f:
                        for row in f.readlines():
                            data["text"].append(row.strip())
                            data["label"].append(folder)
                        f.close()
            except Exception:
                error_file.append(file_path)
                logging.error("Error when reading file {}".format(file_path))
                traceback.print_exc()
    except Exception:
        logging.error("Error when reading folder {}".format(folder_path))

logging.info(f"Error files: [{', '.join(error_file)}]")
df = pd.DataFrame(data, columns=["text", "label"])
logging.info("Num samples: {}".format(len(df)))
df.to_csv(os.path.join(args.output_path, args.output_file), index=False, encoding="utf-8")
logging.info("Saved data to {}".format(os.path.join(args.output_path, args.output_file)))