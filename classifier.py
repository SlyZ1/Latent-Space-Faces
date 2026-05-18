import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.io import ImageReadMode, read_image


from tqdm.auto import tqdm

import os
import sys
from pathlib import Path

PROJECT_DIR = Path.cwd().resolve()
STYLEGAN2_DIR = PROJECT_DIR / 'stylegan2'
if str(STYLEGAN2_DIR) not in sys.path:
    sys.path.insert(0, str(STYLEGAN2_DIR))

from stylegan2 import *
from stylegan2.torch_utils.misc import get_device

def parse_txt_data(path: str | Path) -> tuple[list[str], dict[str, torch.Tensor]]:
    f = open(path, "r")
    _ = f.readline()    # row count

    labels = f.readline().strip().split()
    
    data: dict[str, torch.Tensor] = {}
    for l in f.readlines():
        l = l.strip().split()
        data[l[0]] = torch.Tensor(list(map(float, l[1:])))
    
    f.close()
    return labels, data

class CelebADataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        used_attributes: list[str] = ["Eyeglasses", "Male", "Young"],
        use_landmarks: bool = False,
        transform: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError("The dataset directory was not found")

        self.images_path = self.root / "img_align_celeba"
        if not self.images_path.exists():
            raise FileNotFoundError(f"The image directory was not found at [{self.images_path}]")
        self.attributes_path = self.root / "attributes.txt"
        if not self.attributes_path.exists():
            raise FileNotFoundError(f"The attributes file was not found at [{self.attributes_path}]")
        self.landmarks_path = self.root / "landmarks.txt"
        if not self.landmarks_path.exists():
            raise FileNotFoundError(f"The landmarks file was not found at [{self.landmarks_path}]")

        self.images = list(self.images_path.glob("*.jpg"))
        self.length = len(self.images)

        self.attributes, self.attributes_data = parse_txt_data(self.attributes_path)
        self.used_attributes = [
            self.attributes.index(attr) for attr in used_attributes
        ]

        self.use_landmarks = use_landmarks
        self.landmarks, self.landmarks_data = parse_txt_data(self.landmarks_path)
        self.transform = transform
    
    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):
        image_path = self.images[index]
        img = read_image(str(image_path), mode=ImageReadMode.RGB)
        if self.transform is not None:
            img = self.transform(img)

        labels = torch.Tensor([])
        if len(self.used_attributes) > 0:
            attributes = self.attributes_data[image_path.name][self.used_attributes]
            attributes = (attributes + 1.0) / 2.0
            labels = torch.concat([labels, attributes])
        if self.use_landmarks:
            landmarks = self.landmarks_data[image_path.name]
            labels = torch.concat([labels, landmarks])
        return img, labels

class Classifier(torch.nn.Module):
    def __init__(self, output_size: int = 3, freeze_backbone: bool = True) -> None:
        super().__init__()

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.transform = weights.transforms()

        resnet = torchvision.models.resnet18(weights=weights)
        self.features = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = torch.nn.Flatten(1)
        self.head = torch.nn.Linear(resnet.fc.in_features, output_size)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def trainable_parameters(self):
        return filter(lambda param: param.requires_grad, self.parameters())

    def forward(self, imgs):
        features = self.features(imgs)
        features = self.flatten(features)
        return self.head(features)


if __name__ == "__main__":
    SEED = 42
    _ = torch.manual_seed(SEED)

    MODEL_DIR = Path('models')
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    requested_device = os.environ.get("TORCH_DEVICE")
    device = torch.device(requested_device) if requested_device else get_device()
    print(f"Selected device: {device}")
        
    classifier = Classifier(output_size=3)
    classifier.to(device)

    print("Collecting dataset")
    dataset = CelebADataset("dataset/celeba/", transform=classifier.transform)

    TEST_RATIO = 0.3
    test_set_size = int(round(len(dataset) * TEST_RATIO))
    train_set_size = len(dataset) - test_set_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_set_size, test_set_size])

    EPOCHS = 10
    optimizer = torch.optim.Adam(
        classifier.trainable_parameters(),
        lr=1e-3,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    print("Training")
    MODEL_DIR.mkdir(exist_ok=True)
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    classifier.train()
    for e in range(EPOCHS):
        running_loss = 0

        loop = tqdm(enumerate(train_loader), f"Epoch {e:>3}", total=len(train_loader))
        for i, (img, labels) in loop:
            inputs = img.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = classifier(inputs)
            
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix_str(f"loss: {running_loss/(i+1):.3f}")

        checkpoint_path = MODEL_DIR / "classifier.pt"
        torch.save(classifier.state_dict(), checkpoint_path)
