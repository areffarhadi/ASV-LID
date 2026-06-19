"""
Convert per-file .npy embeddings to .npz format for fusion training.

Input structure (from extract_asv_lid_separate.py):
    emb_base/
        stage1_asv/{train,dev}/speaker_id/language/filename.npy
        stage2_lid/{train,dev}/speaker_id/language/filename.npy
        stage1_asv/metadata.json
        stage2_lid/metadata.json

Output (compatible with train_fusion_dual_path_v12.py):
    output_dir/
        train_asv.npz   — keys: embeddings, filenames, speaker_labels, language_labels
        train_lid.npz
        dev_asv.npz
        dev_lid.npz

Usage:
    python convert_npy_to_npz.py --emb_base ./multilingual_embeddings_stages_tidyvoice --output_dir ./converted_npz
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm


def collect_embeddings(stage_dir, split, speaker_to_id, language_to_id):
    """Walk stage_dir/split/speaker_id/language/*.npy and collect all embeddings."""
    split_dir = os.path.join(stage_dir, split)
    if not os.path.isdir(split_dir):
        print(f"ERROR: Directory not found: {split_dir}")
        sys.exit(1)

    embeddings = []
    filenames = []
    speaker_labels = []
    language_labels = []

    speaker_dirs = sorted(os.listdir(split_dir))
    for spk_id in tqdm(speaker_dirs, desc=f"  {split}"):
        spk_path = os.path.join(split_dir, spk_id)
        if not os.path.isdir(spk_path):
            continue

        spk_label = speaker_to_id.get(spk_id, -1)

        for lang in sorted(os.listdir(spk_path)):
            lang_path = os.path.join(spk_path, lang)
            if not os.path.isdir(lang_path):
                continue

            lang_label = language_to_id.get(lang, -1)

            for npy_file in sorted(os.listdir(lang_path)):
                if not npy_file.endswith('.npy'):
                    continue

                emb = np.load(os.path.join(lang_path, npy_file))
                fn = os.path.splitext(npy_file)[0]  # e.g. "en_40527567"

                embeddings.append(emb)
                filenames.append(fn)
                speaker_labels.append(spk_label)
                language_labels.append(lang_label)

    embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    filenames = np.array(filenames)
    speaker_labels = np.array(speaker_labels, dtype=np.int64)
    language_labels = np.array(language_labels, dtype=np.int64)

    print(f"    {split}: {len(filenames)} samples, {len(set(speaker_labels))} speakers, "
          f"{len(set(language_labels))} languages, emb shape {embeddings.shape}")

    return embeddings, filenames, speaker_labels, language_labels


def convert_stage(stage_dir, stage_name, output_dir):
    """Convert one stage (asv or lid) for both train and dev splits."""
    metadata_path = os.path.join(stage_dir, 'metadata.json')
    if not os.path.isfile(metadata_path):
        print(f"ERROR: metadata.json not found in {stage_dir}")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    speaker_to_id = metadata['speaker_to_id']
    language_to_id = metadata['language_to_id']

    # Convert string keys from JSON to proper types
    speaker_to_id = {k: int(v) for k, v in speaker_to_id.items()}
    language_to_id = {k: int(v) for k, v in language_to_id.items()}

    for split in ['train', 'dev']:
        split_dir = os.path.join(stage_dir, split)
        if not os.path.isdir(split_dir):
            print(f"  Skipping {split} (not found)")
            continue

        embeddings, filenames, speaker_labels, language_labels = collect_embeddings(
            stage_dir, split, speaker_to_id, language_to_id
        )

        out_path = os.path.join(output_dir, f"{split}_{stage_name}.npz")
        np.savez(
            out_path,
            embeddings=embeddings,
            filenames=filenames,
            speaker_labels=speaker_labels,
            language_labels=language_labels,
        )
        print(f"    Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert per-file .npy embeddings to .npz")
    parser.add_argument('--emb_base', type=str, required=True,
                        help='Base directory containing stage1_asv/ and stage2_lid/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for .npz files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for stage_folder, stage_name in [('stage1_asv', 'asv'), ('stage2_lid', 'lid')]:
        stage_dir = os.path.join(args.emb_base, stage_folder)
        if not os.path.isdir(stage_dir):
            print(f"WARNING: {stage_dir} not found, skipping")
            continue

        print(f"\nConverting {stage_folder} → {stage_name}")
        convert_stage(stage_dir, stage_name, args.output_dir)

    print("\nConversion complete!")


if __name__ == '__main__':
    main()
