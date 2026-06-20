from contextlib import nullcontext
from pathlib import Path
import time

import lpips
import torch
import torchvision
import torch.nn.functional as F
from tqdm.auto import tqdm

from model_irse import load_ir_se50
from encoder import (
    FILE_SIZE,
    IMAGENET_NORMALIZE,
    MODELS_DIR,
    N_SCORE_FILES,
    OUT_DIR,
    WEIGHTS,
    get_device,
    load_generator,
    normalize_for_resnet,
)


MODEL_PATH = MODELS_DIR / "perceptual_encoder.pt"

BATCH_SIZE = 2
ACCUMULATION_STEPS = 1
IMAGES_PER_EPOCH = 1024
STEP_SLEEP = 0.05   # seconds to sleep between steps — gives OS breathing room
EPOCHS = 100
HEAD_LR = 1e-3
BACKBONE_LR = 1e-4
HEAD_MIN_LR = 1e-5
BACKBONE_MIN_LR = 1e-6
NUM_WORKERS = 4
MPS_NUM_WORKERS = 0
LOG_EVERY = 1

PIXEL_LOSS_WEIGHT    = 1.0
LPIPS_LOSS_WEIGHT    = 0.8
IDENTITY_LOSS_WEIGHT = 0.1
SYNTHESIS_RESOLUTION = 1024
USE_MPS_AUTOCAST     = False

def generated_to_image(pred: torch.Tensor, size: int = 224) -> torch.Tensor:
    img = ((pred + 1.0) / 2.0).clamp(0.0, 1.0)
    return F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False).contiguous()


def decode_w(delta: torch.Tensor, w_avg: torch.Tensor) -> torch.Tensor:
    """Encoder predicts delta from w_avg; w_pred = w_avg + delta."""
    return w_avg + delta


def synthesize_to_resolution(G: torch.nn.Module, ws: torch.Tensor, resolution: int, **block_kwargs) -> torch.Tensor:
    synthesis = G.synthesis
    if resolution not in synthesis.block_resolutions:
        raise ValueError(f"resolution must be one of {synthesis.block_resolutions}")

    ws = ws.to(torch.float32)
    block_ws = []
    w_idx = 0
    for res in synthesis.block_resolutions:
        block = getattr(synthesis, f"b{res}")
        block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
        w_idx += block.num_conv

    x = img = None
    for res, cur_ws in zip(synthesis.block_resolutions, block_ws):
        block = getattr(synthesis, f"b{res}")
        x, img = block(x, img, cur_ws, force_fp32=True, **block_kwargs)
        if res == resolution:
            break

    if img is None:
        raise RuntimeError("partial synthesis did not produce an image")
    return img


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, out_dir: Path, num_files: int, file_size: int) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.num_files = num_files
        self.file_size = file_size
        self.loaded_file_index = None
        self.loaded_batch = None

    def __len__(self) -> int:
        return self.num_files * self.file_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_index, batch_index = divmod(index, self.file_size)
        if file_index != self.loaded_file_index:
            self.loaded_batch = torch.load(self.out_dir / f"batch_{file_index:03}.pt", map_location="cpu")
            self.loaded_file_index = file_index

        return (
            self.loaded_batch["images"][batch_index].to(torch.float32) / 255.0,
            self.loaded_batch["w"][batch_index].to(torch.float32),
        )


class RandomSubsetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: ImageDataset, num_samples: int) -> None:
        self.data_source = data_source
        self.num_samples = num_samples
        assert num_samples <= len(data_source)

    def __iter__(self):
        indices = torch.randperm(len(self.data_source))[:self.num_samples]
        file_indices = indices // self.data_source.file_size
        return iter(indices[torch.argsort(file_indices)].tolist())

    def __len__(self) -> int:
        return self.num_samples


class ArcFaceIdentityLoss(torch.nn.Module):
    """Cosine identity loss using ArcFace ir_se50, same as pSp paper."""

    def __init__(self) -> None:
        super().__init__()
        self.arcface = load_ir_se50().eval()
        self.arcface.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ir_se50 expects 112×112, [-1, 1]
        pred_r   = F.interpolate(pred,   size=(112, 112), mode="bilinear", align_corners=False) * 2 - 1
        target_r = F.interpolate(target, size=(112, 112), mode="bilinear", align_corners=False) * 2 - 1
        emb_pred = self.arcface(pred_r)
        with torch.no_grad():
            emb_target = self.arcface(target_r)
        return 1 - F.cosine_similarity(emb_pred, emb_target).mean()


class LPIPSLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fn = lpips.LPIPS(net="vgg").eval()
        self.fn.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # LPIPS expects images in [-1, 1]
        pred   = pred   * 2 - 1
        target = target * 2 - 1
        return self.fn(pred, target).mean()


def build_resnet_wplus(num_ws: int, w_dim: int) -> torch.nn.Module:
    resnet = torchvision.models.resnet34(weights=WEIGHTS)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_ws * w_dim)

    # Zero-init the output layer: raw output = 0 → denormalized = w_mean (valid mean face)
    torch.nn.init.zeros_(resnet.fc.weight)
    torch.nn.init.zeros_(resnet.fc.bias)

    for param in resnet.parameters():
        param.requires_grad = False

    for layer in [resnet.layer3, resnet.layer4, resnet.fc]:
        for param in layer.parameters():
            param.requires_grad = True

    return resnet


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    if device.type == "mps" and USE_MPS_AUTOCAST:
        return torch.amp.autocast("mps", dtype=torch.float16)
    return nullcontext()


def no_autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=False)
    return nullcontext()


def synthesis_kwargs(device: torch.device) -> dict:
    kwargs = {"noise_mode": "const"}
    if device.type == "mps":
        kwargs["fused_modconv"] = False
    return kwargs


def dataloader_kwargs(device: torch.device) -> dict:
    num_workers = MPS_NUM_WORKERS if device.type == "mps" else NUM_WORKERS
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs.update(
            {
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
        )
    return kwargs


def freeze_batch_norm(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.requires_grad_(False)


def trainable_parameters(resnet: torch.nn.Module) -> list[dict]:
    backbone_params = []
    head_params = []
    for name, param in resnet.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {"params": backbone_params, "lr": BACKBONE_LR, "min_lr": BACKBONE_MIN_LR},
        {"params": head_params, "lr": HEAD_LR, "min_lr": HEAD_MIN_LR},
    ]


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    lr_lambdas = []
    for group in optimizer.param_groups:
        start_lr = group["lr"]
        min_lr = group["min_lr"]
        lr_lambdas.append(lambda step, start_lr=start_lr, min_lr=min_lr: (min_lr / start_lr) ** (step / total_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)


def load_checkpoint(
    path: Path,
    resnet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> int:
    if not path.exists():
        return 0

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    if (
        config.get("w_encoding") != "delta_from_wavg"
        or config.get("synthesis_resolution") != SYNTHESIS_RESOLUTION
        or config.get("pixel_loss_weight") != PIXEL_LOSS_WEIGHT
        or config.get("identity_loss") != "arcface-ir_se50"
    ):
        print(f"Skipping incompatible checkpoint {path}")
        return 0

    resnet.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"])


def save_checkpoint(
    path: Path,
    resnet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model": resnet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "config": {
                "batch_size": BATCH_SIZE,
                "accumulation_steps": ACCUMULATION_STEPS,
                "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
                "images_per_epoch": IMAGES_PER_EPOCH,
                "epochs": EPOCHS,
                "head_lr": HEAD_LR,
                "backbone_lr": BACKBONE_LR,
                "head_min_lr": HEAD_MIN_LR,
                "backbone_min_lr": BACKBONE_MIN_LR,
                "pixel_loss_weight": PIXEL_LOSS_WEIGHT,
                "lpips_loss_weight": LPIPS_LOSS_WEIGHT,
                "identity_loss_weight": IDENTITY_LOSS_WEIGHT,
                "identity_loss": "arcface-ir_se50",
                "perceptual_loss": "lpips-vgg",
                "synthesis_resolution": SYNTHESIS_RESOLUTION,
                "w_encoding": "delta_from_wavg",
                "unfrozen_layers": ["layer3", "layer4", "fc"],
            },
        },
        path,
    )


def move_images_to_device(imgs: torch.Tensor, device: torch.device) -> torch.Tensor:
    return imgs.to(device, non_blocking=(device.type == "cuda")).contiguous()


def model_to_device(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return model.to(device)


if __name__ == "__main__":
    SEED = 42
    _ = torch.manual_seed(SEED)

    device = get_device()
    print(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    G = load_generator(device)
    w_avg = G.mapping.w_avg.detach().to(device)   # [w_dim] — pSp-style delta anchor

    resnet = model_to_device(build_resnet_wplus(G.num_ws, G.w_dim), device)
    freeze_batch_norm(resnet)

    identity_loss = model_to_device(ArcFaceIdentityLoss(), device).eval()
    lpips_loss    = model_to_device(LPIPSLoss(), device).eval()
    train_set = ImageDataset(OUT_DIR, N_SCORE_FILES, FILE_SIZE)
    train_sampler = RandomSubsetSampler(train_set, IMAGES_PER_EPOCH)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        **dataloader_kwargs(device),
    )

    optimizer = torch.optim.AdamW(trainable_parameters(resnet), weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = (steps_per_epoch + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS
    total_optimizer_steps = EPOCHS * optimizer_steps_per_epoch
    scheduler = make_scheduler(optimizer, total_optimizer_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = load_checkpoint(MODEL_PATH, resnet, optimizer, scheduler, scaler, device)
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch}")

    for e in range(start_epoch, EPOCHS):
        resnet.train()
        freeze_batch_norm(resnet)

        running_loss = torch.zeros((), device=device)
        running_pixel_loss = torch.zeros((), device=device)
        running_lpips_loss = torch.zeros((), device=device)
        running_id_loss = torch.zeros((), device=device)
        steps = 0
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)

        loop = tqdm(train_loader, f"Epoch {e+1:>3}/{EPOCHS:>3}", total=steps_per_epoch)
        for imgs, _ in loop:
            imgs = move_images_to_device(imgs, device)
            # Downscale targets to synthesis resolution to match reconstruction size
            imgs_small = F.interpolate(
                imgs, size=(SYNTHESIS_RESOLUTION, SYNTHESIS_RESOLUTION),
                mode="bilinear", align_corners=False
            )

            with autocast_context(device):
                delta = resnet(normalize_for_resnet(imgs).contiguous())

            with no_autocast_context(device):
                # pSp-style: encoder predicts delta from w_avg, w_pred = w_avg + delta
                delta = delta.float().view(-1, G.num_ws, G.w_dim)
                w_pred = decode_w(delta, w_avg).contiguous()
                reconstructions = synthesize_to_resolution(
                    G, w_pred, SYNTHESIS_RESOLUTION, **synthesis_kwargs(device)
                )
                reconstructions = generated_to_image(reconstructions, size=SYNTHESIS_RESOLUTION)

            pixel_loss = torch.nn.functional.mse_loss(reconstructions, imgs_small)
            with autocast_context(device):
                per_loss = lpips_loss(reconstructions, imgs_small)
                id_loss  = identity_loss(reconstructions, imgs_small)
            loss = (
                PIXEL_LOSS_WEIGHT      * pixel_loss
                + LPIPS_LOSS_WEIGHT    * per_loss
                + IDENTITY_LOSS_WEIGHT * id_loss
            )

            train_loss = loss / ACCUMULATION_STEPS
            should_step = (steps + 1) % ACCUMULATION_STEPS == 0 or (steps + 1) == steps_per_epoch
            if device.type == "cuda":
                scaler.scale(train_loss).backward()
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_steps += 1
            else:
                train_loss.backward()
                if should_step:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    optimizer_steps += 1

            time.sleep(STEP_SLEEP)
            running_loss       = running_loss       + loss.detach()
            running_pixel_loss = running_pixel_loss + pixel_loss.detach()
            running_lpips_loss = running_lpips_loss + per_loss.detach()
            running_id_loss    = running_id_loss    + id_loss.detach()
            steps += 1
            if steps % LOG_EVERY == 0 or steps == steps_per_epoch:
                avg_pixel = running_pixel_loss / steps
                avg_lpips = running_lpips_loss / steps
                avg_id    = running_id_loss / steps
                loop.set_postfix_str(
                    f"loss: {(running_loss/steps).item():.3f}, "
                    f"pixel: {(PIXEL_LOSS_WEIGHT*avg_pixel).item():.3f}, "
                    f"lpips: {(LPIPS_LOSS_WEIGHT*avg_lpips).item():.3f}, "
                    f"id: {(IDENTITY_LOSS_WEIGHT*avg_id).item():.3f}"
                )

        save_checkpoint(MODEL_PATH, resnet, optimizer, scheduler, scaler, e + 1)
