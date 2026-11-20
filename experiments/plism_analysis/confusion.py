"""Export supplied tissue-confusion arrays as numeric CSV without adjustment."""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
import numpy as np

def export_csv(source, output):
    if output.exists():
        raise FileExistsError(output)
    with np.load(source, allow_pickle=False) as saved:
        matrices = saved["matrices"]
        tissues = saved["tissue_labels"].astype(str)
        backbones = saved["backbones"].astype(str)
        pairs = saved["pair_types"].astype(str)
    expected = (len(backbones), 2, len(pairs), len(tissues), len(tissues))
    if matrices.shape != expected or not np.isfinite(matrices).all() or not np.allclose(matrices.sum(axis=-1), 1, rtol=0, atol=1e-12):
        raise ValueError("Invalid paired tissue-confusion probability matrices")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".partial")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["backbone", "method", "pair_type", "query_tissue", "neighbor_tissue", "probability"])
        for bi, backbone in enumerate(backbones):
            for mi, method in enumerate(("RAW", "DELTA")):
                for pi, pair in enumerate(pairs):
                    for qi, query in enumerate(tissues):
                        for ni, neighbor in enumerate(tissues):
                            writer.writerow([backbone, method, pair, query, neighbor, matrices[bi, mi, pi, qi, ni]])
    os.replace(temp, output)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    export_csv(args.input,args.output)

if __name__ == "__main__":
    main()

