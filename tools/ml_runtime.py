"""Small, pinned Ultralytics hooks for local, label-preserving YOLO11 runs."""
import os
from pathlib import Path

# Must precede ultralytics import, including in spawned data-loader workers.
os.environ["YOLO_OFFLINE"] = "true"
os.environ["YOLO_AUTOINSTALL"] = "false"
_config_root = Path(__file__).resolve().parents[1]/".local/m2.2/settings"
_config_dir = _config_root/"Ultralytics"
_config_dir.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(_config_root)
# Native plotting otherwise fetches an optional Arial font, even in offline mode.
# Use Matplotlib's bundled DejaVu Sans through the native font-cache slot.
from importlib.util import find_spec
_font = Path(find_spec("matplotlib").origin).parent/"mpl-data/fonts/ttf/DejaVuSans.ttf"
if not _font.is_file():
    raise RuntimeError("Bundled Matplotlib font is missing")
try:
    (_config_dir/"Arial.ttf").symlink_to(_font)
except FileExistsError:
    pass
os.environ.pop("ULTRALYTICS_API_KEY", None)
os.environ["WANDB_DISABLED"] = "true"

from copy import deepcopy
import json
import shutil
import numpy as np
import torch
import ultralytics
from ultralytics import settings
from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

if ultralytics.__version__ != "8.4.163":
    raise RuntimeError("This transform/trainer hook is verified only with ultralytics==8.4.163")
settings.update({**{k: False for k, v in settings.items() if isinstance(v, bool)}, "api_key": ""})


def preserving_transforms(imgsz, augment, hyp):
    transforms = [LetterBox(new_shape=(imgsz, imgsz), scaleup=False)]
    if augment:
        transforms += [RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                       RandomFlip(p=hyp.flipud, direction="vertical"),
                       RandomFlip(p=hyp.fliplr, direction="horizontal")]
    # No RandomPerspective or Albumentations: no small-object filtering.
    transforms.append(Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=0.0))
    return Compose(transforms)


class PreservingDataset(YOLODataset):
    def get_labels(self):
        original_paths = list(self.im_files)
        labels = super().get_labels()
        if len(labels) != len(original_paths) or {r["im_file"] for r in labels} != set(original_paths):
            raise ValueError("framework filtered images")
        from ultralytics.data.utils import img2label_paths
        for row, path in zip(labels, img2label_paths([r["im_file"] for r in labels])):
            text = Path(path).read_text()
            expected = np.array([[float(v) for v in line.split()] for line in text.splitlines()], dtype=np.float32).reshape(-1, 5)
            actual = np.concatenate([row["cls"], row["bboxes"]], axis=1)
            if expected.shape != actual.shape or sorted(map(tuple, expected)) != sorted(map(tuple, actual)):
                raise ValueError(f"framework altered/deduplicated labels: {path}")
        return labels

    def build_transforms(self, hyp=None):
        return preserving_transforms(self.imgsz, self.augment, hyp)


def build_dataset(args, data, img_path, mode, batch, stride=32):
    return PreservingDataset(img_path=img_path, imgsz=args.imgsz, batch_size=batch,
                             augment=mode == "train", hyp=args, rect=False, cache=False,
                             single_cls=False, stride=stride, pad=0.0, prefix=mode+": ",
                             task="detect", classes=None, data=data, fraction=1.0)


class PreservingValidator(DetectionValidator):
    def __call__(self, trainer=None, model=None):
        result = super().__call__(trainer=trainer, model=model)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir/"effective-evaluation.json").write_text(json.dumps(
            {"args": vars(self.args), "images_seen": self.seen,
             "precision": "FP16" if self.args.quantize == 16 else "FP32",
             "transform_chain": [type(t).__name__ for t in self.dataloader.dataset.transforms.transforms]},
            indent=2, default=str)+"\n")
        return result

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_dataset(self.args, self.data, img_path, mode, batch, self.stride)


def verify_cuda_amp(model, batch):
    """Check a disposable copy of this YOLO11, including real detection loss."""
    if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
        raise RuntimeError("CUDA model required; CPU fallback is forbidden")
    probe = deepcopy(model).float()
    for parameter in probe.parameters():
        parameter.requires_grad_(True)
    probe.eval()
    images = batch["img"][:2].to("cuda").float()/255.0
    with torch.no_grad():
        fp32 = probe(images)[0].float()
        with torch.autocast("cuda", dtype=torch.float16):
            amp = probe(images)[0].float()
    if not torch.isfinite(fp32).all() or not torch.isfinite(amp).all():
        raise RuntimeError("nonfinite FP32/AMP prediction")
    relative_rmse = float((amp-fp32).square().mean().sqrt()/fp32.square().mean().sqrt().clamp_min(1e-9))
    if relative_rmse > 0.02:
        raise RuntimeError(f"AMP prediction relative RMSE too large: {relative_rmse}")
    take = batch["batch_idx"] < len(images)
    sample = {"img": images, "batch_idx": batch["batch_idx"][take].to("cuda"),
              "cls": batch["cls"][take].to("cuda"), "bboxes": batch["bboxes"][take].to("cuda")}
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    before = next(probe.parameters()).detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        loss, _ = probe(sample)
        loss = loss.sum()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite AMP detection loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not all(torch.isfinite(p.grad).all() for p in probe.parameters() if p.grad is not None):
        raise RuntimeError("nonfinite AMP gradients")
    scaler.step(optimizer)
    scaler.update()
    changed = not torch.equal(before, next(probe.parameters()))
    if not changed:
        raise RuntimeError("CUDA optimizer did not update model parameters")
    torch.cuda.synchronize()
    report = {"cuda_allocation": True, "forward_backward": True, "parameters_updated": changed,
              "amp_relative_rmse": relative_rmse, "amp_acceptance_max_relative_rmse": 0.02,
              "loss": float(loss.detach()), "gpu": torch.cuda.get_device_name(0)}
    del probe, optimizer, scaler, images, fp32, amp, loss, before, sample
    torch.cuda.empty_cache()
    return report


class BaselineTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return build_dataset(self.args, self.data, img_path, mode, batch, 32)

    def _setup_train(self):
        # Native check_amp downloads a different architecture (YOLO26n). Bypass
        # only that check and verify this run's YOLO11 on a disposable copy.
        requested_amp = self.args.amp
        self.args.amp = False
        super()._setup_train()
        self.args.amp = requested_amp
        dataset = self.train_loader.dataset
        batch = dataset.collate_fn([dataset[i] for i in range(min(2, len(dataset)))])
        report = verify_cuda_amp(self.model, batch)
        self.amp = bool(requested_amp)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        (self.save_dir/"gpu-check.json").write_text(json.dumps(report, indent=2)+"\n")

    def final_eval(self):
        # Upstream strips optimizer state from best/last during normal finish.
        if self.last.exists():
            shutil.copy2(self.last, self.wdir/"resume.pt")
        self.final_validation_started = True
        super().final_eval()
