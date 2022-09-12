from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from generate import get_parser, get_model
from hgraph.dataset import MoleculeDataset


def cleanup_trace(trace: List[str]) -> List[str]:
    trace_clean = []

    for smiles in trace:
        if not trace_clean or smiles != trace_clean[-1]:
            trace_clean.append(smiles)

    assert len(set(trace_clean)) == len(trace_clean)
    return trace_clean


def reconstruct(model, args, test_smiles):
    dataset = MoleculeDataset(test_smiles, args.vocab, args.atom_vocab, args.batch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x : x[0])

    traces = []
    with torch.no_grad():
        for batch in tqdm(loader):
            results = model.reconstruct(batch)

            for trace in zip(*results):
                trace = cleanup_trace(list(trace))
                traces.append(trace)

    if len(traces) != len(test_smiles):
        raise ValueError("Got less results than expected")

    num_correct = 0
    for smiles, trace in zip(test_smiles, traces):
        print(f"Input: {smiles}")
        print(f"Reconstructed: {trace[-1]}")
        print(f"Correct: {smiles == trace[-1]}")
        print(f"Trace: {trace}")
        print("\n")

        if smiles == trace[-1]:
            num_correct += 1

    print(f"Correct: {num_correct} / {len(test_smiles)} = {num_correct / len(test_smiles)}")


def main():
    parser = get_parser()
    parser.add_argument('--smiles', type=str, required=True)

    args = parser.parse_args()

    test_smiles = [smiles.rstrip() for smiles in open(args.smiles)]
    print(f"Loaded {len(test_smiles)} smiles for testing")

    model = get_model(args)
    reconstruct(model, args, test_smiles)


if __name__ == "__main__":
    main()