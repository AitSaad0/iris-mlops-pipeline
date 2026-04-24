import os
import pandas as pd

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_file(file):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(file.file.read())

    df = pd.read_csv(file_path)

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": df.columns.tolist()
    }