"""
Language Identification (LID) with WPT + W2V-BERT-2.0 + Improved MHFA Head
=============================================================================

This training entrypoint adapts the stronger WPT+MHFA architecture to the same
manifest/data split used in train.sh:
- Flag 1: train
- Flag 2: validation (new speakers)
- Flag 3: crosslingual validation
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from losses import ArcFaceLoss
from main_train_asv import MHFAHeadImproved, WPTW2VBERTMultiLayer

torch.set_default_dtype(torch.float32)


class UnifiedLanguageDataset(Dataset):
    """Dataset for language identification from unified manifest file."""

    def __init__(
        self,
        manifest_file,
        dataset_roots,
        split_flag=None,
        sr=16000,
        audio_len=64600,
        language_to_idx=None,
    ):
        self.samples = []
        self.language_to_idx = language_to_idx if language_to_idx is not None else {}
        self.idx_to_language = {}
        self.sr = sr
        self.audio_len = audio_len
        self.dataset_roots = dataset_roots
        self.split_flag = split_flag
        self._load_manifest(manifest_file)

    def _load_manifest(self, manifest_file):
        language_set = set()
        with open(manifest_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                flag, file_path, language = parts
                if self.split_flag is not None and flag != str(self.split_flag):
                    continue
                language_set.add(language)
                self.samples.append((file_path, language))

        if not self.language_to_idx:
            for idx, language in enumerate(sorted(language_set)):
                self.language_to_idx[language] = idx

        for language, idx in self.language_to_idx.items():
            self.idx_to_language[idx] = language

        print(f"  Loaded {len(self.samples)} samples")
        print(f"  Languages in this set: {sorted(language_set)}")

    def _load_audio(self, file_path):
        full_path = None
        for root in self.dataset_roots:
            candidate = os.path.join(root, file_path)
            if os.path.exists(candidate):
                full_path = candidate
                break

        if full_path is None:
            return torch.zeros(self.audio_len)

        try:
            waveform, sr = torchaudio.load(full_path)
            if sr != self.sr:
                waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze()
            if len(waveform) < self.audio_len:
                pad_amount = self.audio_len - len(waveform)
                waveform = F.pad(waveform, (0, pad_amount))
            else:
                waveform = waveform[: self.audio_len]
            return waveform
        except Exception:
            return torch.zeros(self.audio_len)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, language = self.samples[idx]
        waveform = self._load_audio(file_path)
        language_idx = self.language_to_idx[language]
        return waveform, file_path, torch.tensor(language_idx, dtype=torch.long)


class LanguageIDModelWPTW2VBERTMHFA(nn.Module):
    """LID model with WPT backbone and Improved MHFA pooling head."""

    def __init__(
        self,
        num_languages,
        ssl_model="facebook/w2v-bert-2.0",
        embedding_dim=256,
        num_prompt_tokens=6,
        num_wavelet_tokens=4,
        prompt_dropout=0.1,
        num_heads=8,
        compression_dim=128,
        adapter_bottleneck=128,
        head_dropout=0.1,
        arcface_margin=0.3,
        arcface_scale=30.0,
        peft_mode="deep_prompt",
        use_wavelet=True,
        num_prefix_tokens=None,
    ):
        super().__init__()
        self.num_languages = num_languages
        self.embedding_dim = embedding_dim

        print("\nBuilding Language ID Model (WPT + W2V-BERT + MHFA):")
        print(f"  Number of languages: {num_languages}")
        print(f"  SSL model: {ssl_model}")

        self.wpt_w2vbert = WPTW2VBERTMultiLayer(
            model_dir=ssl_model,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens=num_wavelet_tokens,
            prompt_dim=1024,
            dropout=prompt_dropout,
            peft_mode=peft_mode,
            use_wavelet=use_wavelet,
            num_prefix_tokens=num_prefix_tokens,
        )

        num_layers = self.wpt_w2vbert.config.num_hidden_layers
        self.mhfa_head = MHFAHeadImproved(
            feature_dim=1024,
            num_layers=num_layers,
            num_heads=num_heads,
            compression_dim=compression_dim,
            embedding_dim=embedding_dim,
            adapter_bottleneck=adapter_bottleneck,
            dropout=head_dropout,
        )

        self.arcface = ArcFaceLoss(
            in_features=embedding_dim,
            out_features=num_languages,
            margin=arcface_margin,
            scale=arcface_scale,
        )

    def extract_embedding(self, waveforms, normalize=True):
        layer_features = self.wpt_w2vbert(waveforms)
        emb = self.mhfa_head(layer_features)
        if normalize:
            emb = F.normalize(emb, p=2, dim=1)
        return emb

    def forward(self, waveforms, labels=None):
        emb_unnorm = self.extract_embedding(waveforms, normalize=False)
        emb_norm = F.normalize(emb_unnorm, p=2, dim=1)

        if labels is not None:
            loss, logits = self.arcface(emb_norm, labels)
            return emb_norm, emb_unnorm, logits, loss

        logits = self.arcface.forward_inference(emb_norm)
        return emb_norm, emb_unnorm, logits


def compute_error_rates(y_true, y_score):
    sorted_indices = np.argsort(-y_score)
    y_sorted = y_true[sorted_indices]
    scores_sorted = y_score[sorted_indices]

    n_targets = np.sum(y_true == 1)
    n_non_targets = np.sum(y_true == 0)

    fpr_list = [0.0]
    fnr_list = [1.0]
    threshold_list = [np.inf]
    fp = 0
    fn = n_targets

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            fn -= 1
        else:
            fp += 1
        fpr = fp / n_non_targets if n_non_targets > 0 else 0.0
        fnr = fn / n_targets if n_targets > 0 else 0.0
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        threshold_list.append(scores_sorted[i])

    return np.array(fpr_list), np.array(fnr_list), np.array(threshold_list)


def compute_eer_from_scores(y_true, y_score):
    fpr, fnr, threshold = compute_error_rates(y_true, y_score)
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold_val = threshold[eer_idx]
    return eer, threshold_val


def validate_language_recognition(model, val_loader, device, num_samples=5000):
    print("  Language Recognition (flag=2 EER)...")
    model.eval()
    embeddings_list = []
    languages_list = []

    with torch.no_grad():
        for waveform, _, language_labels in val_loader:
            waveform = waveform.to(device)
            emb_norm, _, _ = model(waveform)
            embeddings_list.append(emb_norm.cpu().numpy())
            languages_list.extend(language_labels.numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    languages = np.array(languages_list)

    language_embeddings = {}
    for lang_idx in np.unique(languages):
        mask = languages == lang_idx
        language_embeddings[lang_idx] = embeddings[mask]

    labels = []
    scores = []
    num_languages = len(language_embeddings)

    for _, embs in language_embeddings.items():
        if len(embs) < 2:
            continue
        n_pairs = min(num_samples // max(2 * num_languages, 1), len(embs) * (len(embs) - 1) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(len(embs), 2, replace=False)
            score = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8)
            labels.append(1)
            scores.append(score)

    lang_indices = list(language_embeddings.keys())
    denom = max(2 * (num_languages * (num_languages - 1) // 2), 1)
    for i in range(len(lang_indices)):
        for j in range(i + 1, len(lang_indices)):
            embs_i = language_embeddings[lang_indices[i]]
            embs_j = language_embeddings[lang_indices[j]]
            n_pairs = min(num_samples // denom, min(len(embs_i), len(embs_j)))
            for _ in range(n_pairs):
                i1 = np.random.randint(0, len(embs_i))
                i2 = np.random.randint(0, len(embs_j))
                score = np.dot(embs_i[i1], embs_j[i2]) / (
                    np.linalg.norm(embs_i[i1]) * np.linalg.norm(embs_j[i2]) + 1e-8
                )
                labels.append(0)
                scores.append(score)

    if not labels:
        print("    Could not generate trial pairs")
        return 1.0, 0.0

    labels = np.array(labels)
    scores = np.array(scores)
    eer, threshold = compute_eer_from_scores(labels, scores)
    print(f"    EER: {100 * eer:.2f}% (threshold: {threshold:.6f})")
    return eer, threshold


def main():
    parser = argparse.ArgumentParser(description="LID Training: WPT + W2V-BERT + MHFA")

    parser.add_argument("--unified_manifest", type=str, required=True)
    parser.add_argument("--dataset_roots", type=str, nargs="+", required=True)
    parser.add_argument("--ssl_model", type=str, default="facebook/w2v-bert-2.0")
    parser.add_argument("--audio_len", type=int, default=64600)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--variance_weight", type=float, default=0.005)

    parser.add_argument("--num_prompt_tokens", type=int, default=6)
    parser.add_argument("--num_wavelet_tokens", type=int, default=4)
    parser.add_argument("--prompt_dropout", type=float, default=0.1)
    # PEFT adaptation mechanism (see src/wpt/peft_wpt.py)
    parser.add_argument("--peft_mode", type=str, default="deep_prompt",
                        choices=["deep_prompt", "shallow_prompt", "prefix"],
                        help="deep_prompt (default = paper) | shallow_prompt | prefix (EXPERIMENTAL)")
    parser.add_argument("--use_wavelet", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--num_prefix_tokens", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--compression_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--adapter_bottleneck", type=int, default=128)
    parser.add_argument("--head_dropout", type=float, default=0.1)

    parser.add_argument("--arcface_margin", type=float, default=0.3)
    parser.add_argument("--arcface_scale", type=float, default=30.0)
    parser.add_argument("--out_fold", type=str, default="./output")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    os.makedirs(args.out_fold, exist_ok=True)
    with open(os.path.join(args.out_fold, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("\nLoading training data (flag=1)...")
    train_dataset = UnifiedLanguageDataset(
        args.unified_manifest,
        args.dataset_roots,
        split_flag=1,
        audio_len=args.audio_len,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    print("\nLoading validation data (flag=2)...")
    val_dataset = UnifiedLanguageDataset(
        args.unified_manifest,
        args.dataset_roots,
        split_flag=2,
        audio_len=args.audio_len,
        language_to_idx=train_dataset.language_to_idx,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print("\nLoading crosslingual validation data (flag=3)...")
    val_cl_dataset = UnifiedLanguageDataset(
        args.unified_manifest,
        args.dataset_roots,
        split_flag=3,
        audio_len=args.audio_len,
        language_to_idx=train_dataset.language_to_idx,
    )
    val_cl_loader = DataLoader(
        val_cl_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    assert train_dataset.language_to_idx == val_dataset.language_to_idx
    assert train_dataset.language_to_idx == val_cl_dataset.language_to_idx

    num_languages = len(train_dataset.language_to_idx)
    model = LanguageIDModelWPTW2VBERTMHFA(
        num_languages=num_languages,
        ssl_model=args.ssl_model,
        embedding_dim=args.embedding_dim,
        num_prompt_tokens=args.num_prompt_tokens,
        num_wavelet_tokens=args.num_wavelet_tokens,
        prompt_dropout=args.prompt_dropout,
        num_heads=args.num_heads,
        compression_dim=args.compression_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        head_dropout=args.head_dropout,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        peft_mode=args.peft_mode,
        use_wavelet=(args.use_wavelet == 'on'),
        num_prefix_tokens=args.num_prefix_tokens,
    ).to(device)

    with open(os.path.join(args.out_fold, "language_mapping.json"), "w") as f:
        json.dump(train_dataset.idx_to_language, f, indent=2)

    trainable_params = []
    trainable_params += [model.wpt_w2vbert.prompt_embeddings]
    trainable_params += [model.wpt_w2vbert.wavelet_prompt_embeddings]
    trainable_params += list(model.mhfa_head.parameters())
    trainable_params += list(model.arcface.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.learning_rate * 0.01
    )

    val_accs = []
    val_cl_accs = []
    lang_rec_eers = []
    best_macro_acc = 0.0

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        model.train()
        model.wpt_w2vbert.model.eval()

        train_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for waveform, _, language_labels in pbar:
            waveform = waveform.to(device)
            language_labels = language_labels.to(device)

            optimizer.zero_grad()
            emb_norm, emb_unnorm, logits, cls_loss = model(waveform, labels=language_labels)
            emb_var = emb_unnorm.var(dim=0).mean()
            variance_loss = torch.clamp(0.5 - emb_var, min=0.0)
            loss = cls_loss + args.variance_weight * variance_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            train_loss += cls_loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += language_labels.size(0)
            correct += (predicted == language_labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{cls_loss.item():.4f}",
                    "var_loss": f"{variance_loss.item():.4f}",
                    "acc": f"{100 * correct / max(total, 1):.1f}%",
                }
            )

        scheduler.step()

        avg_loss = train_loss / max(len(train_loader), 1)
        train_acc = 100 * correct / max(total, 1)
        print(f"  Train: loss={avg_loss:.4f}, acc={train_acc:.1f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        val_predictions = []
        val_labels_list = []

        with torch.no_grad():
            for waveform, _, language_labels in val_loader:
                waveform = waveform.to(device)
                language_labels = language_labels.to(device)
                _, _, logits = model(waveform)
                predicted = torch.argmax(logits, dim=1)
                val_total += language_labels.size(0)
                val_correct += (predicted == language_labels).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(language_labels.cpu().numpy())

        val_acc_micro = 100 * val_correct / max(val_total, 1)
        cm = confusion_matrix(val_labels_list, val_predictions, labels=list(range(num_languages)))
        per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-10)
        val_acc_macro = 100 * per_class_acc.mean()
        val_accs.append((val_acc_micro, val_acc_macro))
        print(f"  Val (flag=2): Micro={val_acc_micro:.1f}%, Macro={val_acc_macro:.1f}%")

        val_cl_correct = 0
        val_cl_total = 0
        val_cl_predictions = []
        val_cl_labels_list = []

        with torch.no_grad():
            for waveform, _, language_labels in val_cl_loader:
                waveform = waveform.to(device)
                language_labels = language_labels.to(device)
                _, _, logits = model(waveform)
                predicted = torch.argmax(logits, dim=1)
                val_cl_total += language_labels.size(0)
                val_cl_correct += (predicted == language_labels).sum().item()
                val_cl_predictions.extend(predicted.cpu().numpy())
                val_cl_labels_list.extend(language_labels.cpu().numpy())

        val_cl_acc_micro = 100 * val_cl_correct / max(val_cl_total, 1)
        unique_langs_cl = sorted(set(val_cl_labels_list))
        cm_cl = confusion_matrix(val_cl_labels_list, val_cl_predictions, labels=unique_langs_cl)
        per_class_acc_cl = np.diag(cm_cl) / (cm_cl.sum(axis=1) + 1e-10)
        val_cl_acc_macro = 100 * per_class_acc_cl.mean()
        val_cl_accs.append((val_cl_acc_micro, val_cl_acc_macro))
        print(f"  Val (flag=3): Micro={val_cl_acc_micro:.1f}%, Macro={val_cl_acc_macro:.1f}%")

        lang_rec_eer, _ = validate_language_recognition(model, val_loader, device)
        lang_rec_eers.append(lang_rec_eer)

        if val_acc_macro > best_macro_acc:
            best_macro_acc = val_acc_macro
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_acc_micro": val_acc_micro,
                    "val_acc_macro": val_acc_macro,
                    "val_cl_acc_micro": val_cl_acc_micro,
                    "val_cl_acc_macro": val_cl_acc_macro,
                    "peft_mode": args.peft_mode,
                    "use_wavelet": (args.use_wavelet == 'on'),
                    "num_prefix_tokens": args.num_prefix_tokens,
                },
                os.path.join(args.out_fold, "best_checkpoint.pt"),
            )
            print(f"  ✓ Best checkpoint saved! (Val Macro: {val_acc_macro:.1f}%)")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_acc_micro": val_acc_micro,
                "val_acc_macro": val_acc_macro,
                "val_cl_acc_micro": val_cl_acc_micro,
                "val_cl_acc_macro": val_cl_acc_macro,
            },
            os.path.join(args.out_fold, f"epoch_{epoch + 1:03d}.pt"),
        )

        with open(os.path.join(args.out_fold, "val_acc.log"), "w") as f:
            for i, (micro, macro) in enumerate(val_accs, start=1):
                f.write(f"Epoch {i}\tMicro: {micro:.2f}%\tMacro: {macro:.2f}%\n")

        with open(os.path.join(args.out_fold, "val_crosslingual_acc.log"), "w") as f:
            for i, (micro, macro) in enumerate(val_cl_accs, start=1):
                f.write(f"Epoch {i}\tMicro: {micro:.2f}%\tMacro: {macro:.2f}%\n")

        with open(os.path.join(args.out_fold, "lang_recognition_eer.log"), "w") as f:
            for i, eer in enumerate(lang_rec_eers, start=1):
                f.write(f"Epoch {i}\tEER: {100 * eer:.2f}%\n")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation macro-averaged accuracy: {best_macro_acc:.1f}%")


if __name__ == "__main__":
    main()
