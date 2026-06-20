import pickle
import sys
import warnings
from pathlib import Path

import torch
import torchvision
from tqdm.auto import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
STYLEGAN2_DIR = PROJECT_DIR / "stylegan2"
if str(STYLEGAN2_DIR) not in sys.path:
    sys.path.insert(0, str(STYLEGAN2_DIR))

from stylegan2.torch_utils import custom_ops
from stylegan2.torch_utils.misc import get_device


custom_ops.force_disable_problematic_cuda_kernels = False
custom_ops.verbosity = "brief"

MODELS_DIR = PROJECT_DIR / "models"
OUT_DIR = PROJECT_DIR / "out"
MODEL_PATH = MODELS_DIR / "encoder.pt"
GAN_PATH = MODELS_DIR / "ffhq.pkl"
W_STATS_PATH = OUT_DIR / "w_stats.pt"

WEIGHTS = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
IMAGENET_TRANSFORM = WEIGHTS.transforms()
IMAGENET_NORMALIZE = torchvision.transforms.Normalize(
    mean=IMAGENET_TRANSFORM.mean,
    std=IMAGENET_TRANSFORM.std,
)

BATCH_SIZE = 8
FILE_SIZE = 1024
N_SCORE_FILES = 64
N_SCORED_IMAGES = FILE_SIZE * N_SCORE_FILES
IMAGES_PER_EPOCH = 8192
EPOCHS = 50
LR = 1e-2
MIN_LR = 1e-4
NUM_WORKERS = 4


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


def preprocess_prediction(pred: torch.Tensor, width: int = 224) -> torch.Tensor:
    img = (pred + 1.0) / 2.0
    img = torch.clip(img, 0.0, 1.0)
    img = torchvision.transforms.Resize((width, width), antialias=False)(img)
    return img


def prediction_to_img(pred: torch.Tensor):
    img = (pred + 1.0) / 2.0
    img = torch.clip(img, 0.0, 1.0)
    img = img.permute(1, 2, 0)
    return img.cpu().numpy()


def normalize_for_resnet(imgs: torch.Tensor) -> torch.Tensor:
    return IMAGENET_NORMALIZE(imgs)


def load_or_calculate_w_stats(out_dir: Path, num_files: int) -> tuple[torch.Tensor, torch.Tensor]:
    if W_STATS_PATH.exists():
        stats = torch.load(W_STATS_PATH, map_location="cpu")
        if stats["num_files"] == num_files:
            return stats["mean"], stats["std"]

    total = None
    squared_total = None
    count = 0
    for i in tqdm(range(num_files), desc="Calculating W statistics"):
        w = torch.load(out_dir / f"batch_{i:03}.pt", map_location="cpu")["w"].to(torch.float64)
        if total is None:
            total = torch.zeros(w.shape[1], dtype=torch.float64)
            squared_total = torch.zeros(w.shape[1], dtype=torch.float64)
        total += w.sum(dim=0)
        squared_total += (w * w).sum(dim=0)
        count += len(w)

    assert total is not None and squared_total is not None
    mean = (total / count).to(torch.float32)
    variance = squared_total / count - mean.to(torch.float64) ** 2
    std = variance.clamp_min(0.0).sqrt().clamp_min(1e-6).to(torch.float32)
    torch.save({"mean": mean, "std": std, "num_files": num_files}, W_STATS_PATH)
    return mean, std


def normalize_w(w: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (w - mean) / std


def build_resnet(w_dim: int) -> torch.nn.Module:
    resnet = torchvision.models.resnet34(weights=WEIGHTS)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, w_dim)

    for param in resnet.parameters():
        param.requires_grad = False

    for layer in [resnet.layer4, resnet.fc]:
        for param in layer.parameters():
            param.requires_grad = True

    return resnet


def load_generator(device: torch.device):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed.*",
            category=Warning,
        )
        with open(GAN_PATH, "rb") as f:
            G = pickle.load(f)["G_ema"].to(device)
    return G.eval().requires_grad_(False)


def load_model_state(resnet: torch.nn.Module, path: Path, device: torch.device) -> int:
    if not path.exists():
        return 0

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    try:
        resnet.load_state_dict(state_dict)
    except RuntimeError as error:
        print(f"Skipping incompatible checkpoint {path}: {error}")
        return 0
    return int(checkpoint.get("epoch", 0)) if "model" in checkpoint else 0


def save_checkpoint(
    path: Path,
    resnet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model": resnet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "config": {
                "n_scored_images": N_SCORED_IMAGES,
                "n_score_files": N_SCORE_FILES,
                "file_size": FILE_SIZE,
                "images_per_epoch": IMAGES_PER_EPOCH,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "min_lr": MIN_LR,
                "output_dim": resnet.fc.out_features,
                "normalize_w_loss": True,
            },
        },
        path,
    )


if __name__ == "__main__":
    SEED = 42
    _ = torch.manual_seed(SEED)

    device = get_device()
    print(device)

    G = load_generator(device)
    w_mean, w_std = load_or_calculate_w_stats(OUT_DIR, N_SCORE_FILES)
    w_mean = w_mean.to(device)
    w_std = w_std.to(device)

    resnet = build_resnet(G.w_dim).to(device)
    train_set = ImageDataset(OUT_DIR, N_SCORE_FILES, FILE_SIZE)
    train_sampler = RandomSubsetSampler(train_set, IMAGES_PER_EPOCH)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=(NUM_WORKERS > 0),
    )
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, resnet.parameters()), lr=LR)
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=(MIN_LR / LR) ** (1 / total_steps),
    )
    criterion = torch.nn.MSELoss()

    start_epoch = load_model_state(resnet, MODEL_PATH, device)
    if start_epoch > 0:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if "model" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resuming from epoch {start_epoch}")

    resnet.train()
    for e in range(start_epoch, EPOCHS):
        running_loss = 0.0
        steps = 0

        loop = tqdm(train_loader, f"Epoch {e+1:>3}/{EPOCHS:>3}", total=steps_per_epoch)
        for imgs, w_gt in loop:
            imgs = imgs.to(device)
            w_gt = w_gt.to(device)

            w_pred = resnet(normalize_for_resnet(imgs))
            loss = criterion(normalize_w(w_pred, w_mean, w_std), normalize_w(w_gt, w_mean, w_std))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            steps += 1
            loop.set_postfix_str(f"loss: {running_loss/steps:.3f}")

        save_checkpoint(MODEL_PATH, resnet, optimizer, scheduler, e + 1)
