import argparse
import os
import warnings
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm
from torch.cuda.amp import GradScaler

# Silence torchvision video API deprecation warnings
warnings.filterwarnings(
    "ignore",
    message=r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.io.*",
)

from src.config import DEFAULT_CONFIG, Config
from src.data.dataset import MultiModalEmotionDataset
from src.models.encoders import EmotionModel
from src.utils.device import autocast_enabled, get_device


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description="Train multimodal emotion classifier")
    parser.add_argument("--data-root", default=cfg.data.data_root)
    parser.add_argument("--text-map", type=Path, default=None, help="Optional JSON map: stem -> text")
    parser.add_argument("--num-frames", type=int, default=cfg.data.num_frames)
    parser.add_argument(
        "--frame-strategy",
        choices=["random", "uniform"],
        default=cfg.data.frame_strategy,
    )
    parser.add_argument("--sample-rate", type=int, default=cfg.data.sample_rate)
    parser.add_argument("--batch-size", type=int, default=cfg.train.batch_size)
    parser.add_argument("--num-workers", type=int, default=cfg.train.num_workers)
    parser.add_argument("--epochs", type=int, default=cfg.train.epochs)
    parser.add_argument("--lr", type=float, default=cfg.train.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.train.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.train.grad_clip)
    parser.add_argument(
        "--intensity-loss-weight", type=float, default=cfg.train.intensity_loss_weight
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable automatic mixed precision"
    )
    parser.add_argument("--freeze-video", action="store_true")
    parser.add_argument("--freeze-audio", action="store_true")
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--output-dir", default=cfg.train.output_dir)
    parser.add_argument("--train-split", type=float, default=cfg.data.train_split)
    parser.add_argument("--max-text-length", type=int, default=cfg.data.max_text_length)
    parser.add_argument("--tokenizer", default=cfg.model.text_encoder)
    parser.add_argument(
        "--cached-dataset",
        type=Path,
        action="append",
        default=None,
        help="Path(s) to pre-decoded dataset .pt files (pass multiple times for shards)",
    )
    parser.add_argument(
        "--pre-decode",
        action=argparse.BooleanOptionalAction,
        default=cfg.train.pre_decode,
        help="Decode dataset once before training to remove per-epoch IO/ffmpeg overhead",
    )
    parser.add_argument(
        "--pre-decode-device",
        choices=["auto", "cpu", "cuda"],
        default=cfg.train.pre_decode_device,
        help="Where to place cached tensors when pre-decode is enabled",
    )
    parser.add_argument("--zero-video", action="store_true", help="Ablation: zero out video inputs")
    parser.add_argument("--zero-audio", action="store_true", help="Ablation: zero out audio inputs")
    parser.add_argument("--zero-text", action="store_true", help="Ablation: drop text inputs (set to None)")
    return parser.parse_args()


def prepare_config(args: argparse.Namespace) -> Config:
    cfg = DEFAULT_CONFIG
    cfg.data.data_root = args.data_root
    cfg.data.num_frames = args.num_frames
    cfg.data.frame_strategy = args.frame_strategy
    cfg.data.sample_rate = args.sample_rate
    cfg.data.train_split = args.train_split
    cfg.data.max_text_length = args.max_text_length
    cfg.data.text_map = args.text_map

    cfg.model.text_encoder = args.tokenizer
    cfg.model.freeze_video = args.freeze_video
    cfg.model.freeze_audio = args.freeze_audio
    cfg.model.freeze_text = args.freeze_text

    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.grad_clip = args.grad_clip
    cfg.train.intensity_loss_weight = args.intensity_loss_weight
    cfg.train.amp = not args.no_amp
    cfg.train.output_dir = args.output_dir
    cfg.train.pre_decode = args.pre_decode
    cfg.train.pre_decode_device = args.pre_decode_device
    return cfg


def _single_item_collate(batch):
    return batch[0]


def _tokenize_once(tokenizer, text: str, max_len: int):
    if tokenizer is None:
        return None
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in encoded.items()}


class PreDecodedDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def predecoded_collate(batch):
    videos = torch.stack([b["video"] for b in batch])
    audios = [b["audio"] for b in batch]
    audio_lens = torch.tensor(
        [a.shape[0] for a in audios], dtype=torch.long, device=videos.device
    )
    audios = pad_sequence(audios, batch_first=True)
    labels = torch.stack([b["labels"] for b in batch])

    text_inputs = None
    if batch[0]["text_inputs"] is not None:
        text_inputs = {
            k: torch.stack([sample["text_inputs"][k] for sample in batch])
            for k in batch[0]["text_inputs"].keys()
        }

    return {
        "video": videos,
        "audio": audios,
        "audio_lens": audio_lens,
        "text_inputs": text_inputs,
        "labels": labels,
    }


def _sort_indices_by_stem(stems):
    def stem_key(s):
        try:
            return (0, int(s))
        except (TypeError, ValueError):
            return (1, str(s))

    paired = [(stem_key(stems[i]), i) for i in range(len(stems))]
    paired.sort(key=lambda x: x[0])
    return [idx for _, idx in paired]


def _split_by_stem(dataset, stems, train_ratio: float):
    sorted_indices = _sort_indices_by_stem(stems)
    train_len = int(len(sorted_indices) * train_ratio)
    train_idx = sorted_indices[:train_len]
    val_idx = sorted_indices[train_len:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def _stratified_split(dataset, labels, stems, train_ratio: float):
    by_label = defaultdict(list)
    for idx, (lab, stem) in enumerate(zip(labels, stems)):
        by_label[int(lab)].append((stem, idx))

    train_indices = []
    val_indices = []
    for lab, items in by_label.items():
        # sort within label by stem for determinism
        items.sort(key=lambda x: (0, int(x[0])) if str(x[0]).isdigit() else (1, str(x[0])))
        n = len(items)
        n_train = int(n * train_ratio)
        lab_train = [idx for _, idx in items[:n_train]]
        lab_val = [idx for _, idx in items[n_train:]]
        train_indices.extend(lab_train)
        val_indices.extend(lab_val)

    # keep overall order deterministic
    train_indices.sort()
    val_indices.sort()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def resolve_predecode_device(requested: str, train_device: torch.device) -> torch.device:
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA pre-decode requested but CUDA is not available; using CPU instead.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if train_device.type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def predecode_split(split, base_dataset, device: torch.device, num_workers: int, desc: str):
    loader = DataLoader(
        split,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_single_item_collate,
        pin_memory=False,
    )
    cached = []
    for sample in tqdm(loader, desc=desc):
        text_inputs = _tokenize_once(base_dataset.tokenizer, sample["text"], base_dataset.max_text_length)
        if text_inputs is not None:
            text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}

        cached.append(
            {
                "video": sample["video"].to(device, non_blocking=True),
                "audio": sample["audio"].to(device, non_blocking=True),
                "labels": torch.tensor(sample["label"], dtype=torch.long, device=device),
                "text_inputs": text_inputs,
            }
        )
    return cached


def move_text_inputs(text_inputs, device):
    if text_inputs is None:
        return None
    return {k: v.to(device) for k, v in text_inputs.items()}


def save_checkpoint(model: EmotionModel, output_dir: str, tag: str):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{tag}.pt"
    torch.save(model.state_dict(), path)
    return path


def main():
    args = parse_args()
    cfg = prepare_config(args)

    device = get_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        print(f"Using device: {device} (name={name}, total={torch.cuda.device_count()}, available={torch.cuda.is_available()})")
    else:
        print(f"Using device: {device} (cuda_available={torch.cuda.is_available()})")

    # MPS 的 float16 训练容易出现 NaN；默认关闭 AMP，改用全精度
    if device.type == "mps" and cfg.train.amp:
        cfg.train.amp = False
        print("AMP is disabled on MPS to avoid NaN issues; running in float32.")

    # Suppress noisy video deprecation warning from torchvision (multiple matches to be safe)
    warnings.filterwarnings(
        "ignore",
        message=r"The video decoding and encoding capabilities of torchvision are deprecated",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"torchvision\.io",
    )

    pin_memory = device.type == "cuda"  # MPS/CPU 不支持 pin_memory，避免无效警告
    loader_workers = cfg.train.num_workers
    collate_fn = None

    cached_paths_arg = args.cached_dataset if args.cached_dataset is not None else []
    if cached_paths_arg:
        paths: List[Path] = []
        for p in cached_paths_arg:
            p = p.expanduser()
            if p.is_dir():
                # Load all .pt files in directory, sorted by natural numeric order
                pt_files = sorted(p.glob("*.pt"), key=lambda x: [int(t) if t.isdigit() else t for t in x.stem.split('_')])
                paths.extend(pt_files)
            else:
                paths.append(p)
        if not paths:
            raise FileNotFoundError("No cached dataset files found in provided paths")
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Cached dataset not found: {p}")
        print("Loading pre-decoded dataset(s):")
        for p in paths:
            print(f"  - {p}")

        all_samples = []
        cached_cfg = {}
        for idx, p in enumerate(paths):
            cache_obj = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(cache_obj, dict) and "samples" in cache_obj:
                samples_part = cache_obj["samples"]
                cfg_part = cache_obj.get("config", {})
            else:
                samples_part = cache_obj
                cfg_part = {}
            all_samples.extend(samples_part)
            if idx == 0:
                cached_cfg = cfg_part

        # Warn if core decoding params differ (use first shard as reference)
        if cached_cfg:
            mismatches = []
            if cached_cfg.get("num_frames") and cached_cfg["num_frames"] != cfg.data.num_frames:
                mismatches.append(f"num_frames cached={cached_cfg['num_frames']} current={cfg.data.num_frames}")
            if cached_cfg.get("sample_rate") and cached_cfg["sample_rate"] != cfg.data.sample_rate:
                mismatches.append(f"sample_rate cached={cached_cfg['sample_rate']} current={cfg.data.sample_rate}")
            if mismatches:
                print("Warning: cached dataset params differ: " + "; ".join(mismatches))

        base_dataset = PreDecodedDataset(all_samples)
        stems = [s.get("stem") for s in all_samples]
        labels = [int(s.get("labels")) for s in all_samples]
        train_dataset, val_dataset = _stratified_split(base_dataset, labels, stems, cfg.data.train_split)
        collate_fn = predecoded_collate
        loader_workers = 0
        pin_memory = pin_memory  # keep pinning when training on CUDA
        cfg.train.pre_decode = False  # avoid double predecode
    else:
        dataset = MultiModalEmotionDataset(
            root=cfg.data.data_root,
            num_frames=cfg.data.num_frames,
            frame_strategy=cfg.data.frame_strategy,
            sample_rate=cfg.data.sample_rate,
            tokenizer_name=cfg.model.text_encoder,
            max_text_length=cfg.data.max_text_length,
            text_map_path=cfg.data.text_map,
        )

        stems = [s["stem"] for s in dataset.samples]
        labels = [s["label"] for s in dataset.samples]
        train_set, val_set = _stratified_split(dataset, labels, stems, cfg.data.train_split)

        train_dataset = train_set
        val_dataset = val_set
        collate_fn = dataset.collate_fn

        if cfg.train.pre_decode:
            predecode_device = resolve_predecode_device(cfg.train.pre_decode_device, device)
            print(f"Pre-decoding dataset to {predecode_device} before training...")
            train_cached = predecode_split(
                train_set,
                dataset,
                predecode_device,
                cfg.train.num_workers,
                desc="Pre-decode train",
            )
            val_cached = predecode_split(
                val_set,
                dataset,
                predecode_device,
                cfg.train.num_workers,
                desc="Pre-decode val",
            )
            train_dataset = PreDecodedDataset(train_cached)
            val_dataset = PreDecodedDataset(val_cached)
            collate_fn = predecoded_collate
            loader_workers = 0  # Cached tensors do not need worker pools; avoids CUDA tensors in workers
            pin_memory = pin_memory and predecode_device.type == "cpu"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=loader_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=loader_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    model = EmotionModel(
        num_classes=7,
        video_name=cfg.model.video_encoder,
        audio_name=cfg.model.audio_encoder,
        text_name=cfg.model.text_encoder,
        fusion_hidden=cfg.model.fusion_hidden,
        dropout=cfg.model.dropout,
        freeze_video=cfg.model.freeze_video,
        freeze_audio=cfg.model.freeze_audio,
        freeze_text=cfg.model.freeze_text,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    amp_enabled = cfg.train.amp and autocast_enabled(device)
    scaler_enabled = amp_enabled and device.type == "cuda"  # GradScaler only meaningful on CUDA
    scaler = GradScaler(enabled=scaler_enabled)

    best_acc = 0.0

    train_acc_hist = []
    val_acc_hist = []
    train_loss_hist = []
    val_loss_hist = []

    for epoch in range(cfg.train.epochs):
        model.train()
        train_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs} [train]"
        )
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0
        train_conf_sum = 0.0
        for step, batch in enumerate(train_bar):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            text_inputs = move_text_inputs(batch["text_inputs"], device)

            if args.zero_video:
                video = torch.zeros_like(video)
            if args.zero_audio:
                audio = torch.zeros_like(audio)
            if args.zero_text:
                text_inputs = None

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits, intensity = model(video, audio, text_inputs)
                cls_loss = ce_loss(logits, labels)
                num_classes = logits.shape[1]
                target_intensity = labels.float() / max(1, num_classes - 1)
                inten_loss = mse_loss(intensity, target_intensity)
                loss = cls_loss + cfg.train.intensity_loss_weight * inten_loss

            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            probs = torch.softmax(logits, dim=1)
            train_conf_sum += probs[torch.arange(labels.size(0)), preds].sum().item()

            if step % cfg.train.log_every == 0:
                train_bar.set_postfix(
                    {
                        "loss": loss.item(),
                        "cls": cls_loss.item(),
                        "inten": inten_loss.item(),
                    }
                )

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_conf_sum = 0.0
        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{cfg.train.epochs} [val]"
            ):
                video = batch["video"].to(device)
                audio = batch["audio"].to(device)
                labels = batch["labels"].to(device)
                text_inputs = move_text_inputs(batch["text_inputs"], device)

                if args.zero_video:
                    video = torch.zeros_like(video)
                if args.zero_audio:
                    audio = torch.zeros_like(audio)
                if args.zero_text:
                    text_inputs = None
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    logits, intensity = model(video, audio, text_inputs)
                    cls_loss = ce_loss(logits, labels)
                    num_classes = logits.shape[1]
                    target_intensity = labels.float() / max(1, num_classes - 1)
                    inten_loss = mse_loss(intensity, target_intensity)
                    loss = cls_loss + cfg.train.intensity_loss_weight * inten_loss
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                probs = torch.softmax(logits, dim=1)
                val_conf_sum += probs[torch.arange(labels.size(0)), preds].sum().item()

            train_loss_avg = train_loss_sum / max(1, train_total)
            train_acc = train_correct / max(1, train_total)
            train_conf = train_conf_sum / max(1, train_total)
            val_loss = val_loss / max(1, total)
            val_acc = correct / max(1, total)
            val_conf = val_conf_sum / max(1, total)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss_avg:.4f} train_acc={train_acc:.4f} "
                f"train_conf={train_conf:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_conf={val_conf:.4f}"
            )

            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)
            train_loss_hist.append(train_loss_avg)
            val_loss_hist.append(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = save_checkpoint(model, cfg.train.output_dir, "best")
            print(f"Saved best checkpoint to {ckpt_path}")

    final_ckpt = save_checkpoint(model, cfg.train.output_dir, "last")
    print(f"Training done. Last checkpoint: {final_ckpt}")

    # Plot accuracy/loss curves
    try:
        import matplotlib.pyplot as plt

        out_dir = Path(cfg.train.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        epochs_range = range(1, cfg.train.epochs + 1)

        plt.figure(figsize=(8, 4))
        plt.plot(epochs_range, train_acc_hist, label="train_acc")
        plt.plot(epochs_range, val_acc_hist, label="val_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "accuracy_curve.png")
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(epochs_range, train_loss_hist, label="train_loss")
        plt.plot(epochs_range, val_loss_hist, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png")
        plt.close()
        print(f"Saved curves to {out_dir}/accuracy_curve.png and loss_curve.png")
    except ImportError:
        print("matplotlib not installed; skipping curve plotting")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
