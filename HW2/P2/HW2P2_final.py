import torch
from torchsummary import summary
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics as mt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import glob
import torch.nn as nn
import wandb
import matplotlib.pyplot as plt
from pytorch_metric_learning import samplers
import csv
from torch.utils.data import default_collate
from torchvision.transforms import v2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
root = "/home/yash/content/data/11-785-f24-hw2p2-verification"
checkpoint_dir = "/home/yash/checkpoint"

print("Device: ", DEVICE)

config = {
    "data_dir": f"{root}/cls_data",
    "data_ver_dir": f"{root}/ver_data",
    "checkpoint_dir": checkpoint_dir,
    "batch_size": 128,
    "lr": 0.01,
    "epochs": 30,
    "scheduler": {
        "name": "CosineAnnealingWithWarmRestarts",
        "T_0": 10,
        "T_mult": 2,
        "eta_min": 1e-5,
    },
    "optimizer": {"name": "SGD", "momentum": 0.9, "weight_decay": 5e-4},
}

data_dir = config["data_dir"]
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "dev")


class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith(".csv"):
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, "r") as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(" "))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        img_path1, img_path2, match = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1))
        img2 = Image.open(os.path.join(self.data_dir, img_path2))
        return self.transform(img1), self.transform(img2), int(match)


class TestImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.pairs = []
        if csv_file.endswith(".csv"):
            with open(csv_file, "r") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    else:
                        self.pairs.append(row)
        else:
            with open(csv_file, "r") as f:
                for line in f.readlines():
                    self.pairs.append(line.strip().split(" "))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        img_path1, img_path2 = self.pairs[idx]
        img1 = Image.open(os.path.join(self.data_dir, img_path1))
        img2 = Image.open(os.path.join(self.data_dir, img_path2))
        return self.transform(img1), self.transform(img2)


# train transforms
train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(112),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(5),
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
        ),
        torchvision.transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomErasing(
            p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)
        ),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# val transforms
val_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(112),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# get datasets
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transforms)

# cutmix
cutmix = v2.CutMix(alpha=0.8, num_classes=len(train_dataset.classes))


def collate_fn(batch):
    return cutmix(*default_collate(batch))


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    pin_memory=True,
    num_workers=8,
    sampler=None,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
)

data_dir = config["data_ver_dir"]

pair_dataset = ImagePairDataset(
    data_dir, csv_file=f"{root}/val_pairs.txt", transform=val_transforms
)
pair_dataloader = torch.utils.data.DataLoader(
    pair_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)

test_pair_dataset = TestImagePairDataset(
    data_dir, csv_file=f"{root}/test_pairs.txt", transform=val_transforms
)
test_pair_dataloader = torch.utils.data.DataLoader(
    test_pair_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    pin_memory=True,
    num_workers=4,
)

print("Number of classes    : ", len(train_dataset.classes))
print("No. of train images  : ", train_dataset.__len__())
print("Shape of image       : ", train_dataset[0][0].shape)
print("Batch size           : ", config["batch_size"])
print("Train batches        : ", train_loader.__len__())
print("Val batches          : ", val_loader.__len__())


# Model Architecture
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        feats = self.avgpool(out)
        feats = feats.view(feats.size(0), -1)
        out = self.linear(feats)
        return {"feats": feats, "out": out}


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=len(train_dataset.classes))


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [
        correct[: min(k, maxk)].reshape(-1).float().sum(0) * 100.0 / batch_size
        for k in topk
    ]


def get_ver_metrics(labels, scores, FPRs):
    # eer and auc
    fpr, tpr, _ = mt.roc_curve(labels, scores, pos_label=1)
    roc_curve = interp1d(fpr, tpr)
    EER = 100.0 * brentq(lambda x: 1.0 - x - roc_curve(x), 0.0, 1.0)
    AUC = 100.0 * mt.auc(fpr, tpr)

    # get acc
    tnr = 1.0 - fpr
    pos_num = labels.count(1)
    neg_num = labels.count(0)
    ACC = 100.0 * max(tpr * pos_num + tnr * neg_num) / len(labels)

    # TPR @ FPR
    if isinstance(FPRs, list):
        TPRs = [
            ("TPR@FPR={}".format(FPR), 100.0 * roc_curve(float(FPR))) for FPR in FPRs
        ]
    else:
        TPRs = []

    return {
        "ACC": ACC,
        "EER": EER,
        "AUC": AUC,
        "TPRs": TPRs,
    }


model = ResNet18().to(DEVICE)
summary(model, (3, 112, 112))

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config["lr"],
    momentum=config["optimizer"]["momentum"],
    weight_decay=config["optimizer"]["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=config["scheduler"]["T_0"],
    T_mult=config["scheduler"]["T_mult"],
    eta_min=config["scheduler"]["eta_min"],
)

scaler = torch.amp.GradScaler("cuda")


# Train and Validation Function
def train_epoch(model, dataloader, optimizer, lr_scheduler, scaler, device, config):

    model.train()

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    # Progress Bar
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        leave=False,
        position=0,
        desc="Train",
        ncols=5,
    )

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()  # Zero gradients

        # send to cuda
        images = images.to(device, non_blocking=True)
        if isinstance(labels, (tuple, list)):
            targets1, targets2, lam = labels
            labels = (targets1.to(device), targets2.to(device), lam)
        else:
            labels = labels.to(device, non_blocking=True)

        # forward
        with torch.amp.autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs["out"], labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # metrics
        loss_m.update(loss.item())
        if "feats" in outputs:
            acc = accuracy(outputs["out"], labels.argmax(dim=1))[0].item()
        else:
            acc = 0.0
        acc_m.update(acc)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            # acc         = "{:.04f}%".format(100*accuracy),
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
            lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
        )

        batch_bar.update()  # Update tqdm bar

    if lr_scheduler is not None:
        lr_scheduler.step()

    batch_bar.close()

    return acc_m.avg, loss_m.avg


@torch.no_grad()
def valid_epoch_cls(model, dataloader, device, config):
    model.eval()
    batch_bar = tqdm(
        total=len(dataloader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Val Cls.",
        ncols=5,
    )

    # metric meters
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs["out"], labels)

        # metrics
        acc = accuracy(outputs["out"], labels)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(
            acc="{:.04f}% ({:.04f})".format(acc, acc_m.avg),
            loss="{:.04f} ({:.04f})".format(loss.item(), loss_m.avg),
        )

        batch_bar.update()

    batch_bar.close()
    return acc_m.avg, loss_m.avg


gc.collect()
torch.cuda.empty_cache()


# # Verification Task
def valid_epoch_ver(model, pair_data_loader, device, config):
    model.eval()
    scores = []
    match_labels = []
    batch_bar = tqdm(
        total=len(pair_data_loader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Val Veri.",
    )
    for i, (images1, images2, labels) in enumerate(pair_data_loader):
        images = torch.cat([images1, images2], dim=0).to(device)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.append(similarity.cpu().numpy())
        match_labels.append(labels.cpu().numpy())
        batch_bar.update()

    scores = np.concatenate(scores)
    match_labels = np.concatenate(match_labels)

    FPRs = ["1e-4", "5e-4", "1e-3", "5e-3", "5e-2"]
    metric_dict = get_ver_metrics(match_labels.tolist(), scores.tolist(), FPRs)
    print(metric_dict)

    return metric_dict["ACC"], metric_dict["EER"]


def save_model(model, optimizer, scheduler, metrics, epoch, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metric": metrics,
            "epoch": epoch,
        },
        path,
    )


def load_model(model, optimizer=None, scheduler=None, path="./checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer = None
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        scheduler = None
    epoch = checkpoint["epoch"]
    metrics = checkpoint["metric"]
    return model, optimizer, scheduler, epoch, metrics


# WandB
# wandb.login(key="") # API Key is in your wandb account, under settings (wandb.ai/settings)

# run = wandb.init(
#     name = "name", ## Wandb creates random run names if you skip this field
#     reinit = True, ### Allows reinitalizing runs when you re-run this cell
#     # run_id = ### Insert specific run id here if you want to resume a previous run
#     # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
#     project = "hw2p2", ### Project should be created in your wandb account
#     config = config ### Wandb Config for your run
# )

e = 0
best_valid_cls_acc = 0.0
eval_cls = True
best_valid_ret_acc = 0.0
for epoch in range(e, config["epochs"]):
    # epoch
    print("\nEpoch {}/{}".format(epoch + 1, config["epochs"]))

    # train
    train_cls_acc, train_loss = train_epoch(
        model, train_loader, optimizer, scheduler, scaler, DEVICE, config
    )
    curr_lr = float(optimizer.param_groups[0]["lr"])
    print(
        "\nEpoch {}/{}: \nTrain Cls. Acc {:.04f}%\t Train Cls. Loss {:.04f}\t Learning Rate {:.04f}".format(
            epoch + 1, config["epochs"], train_cls_acc, train_loss, curr_lr
        )
    )
    metrics = {
        "train_cls_acc": train_cls_acc,
        "train_loss": train_loss,
    }
    # classification validation
    if eval_cls:
        valid_cls_acc, valid_loss = valid_epoch_cls(model, val_loader, DEVICE, config)
        print(
            "Val Cls. Acc {:.04f}%\t Val Cls. Loss {:.04f}".format(
                valid_cls_acc, valid_loss
            )
        )
        metrics.update(
            {
                "valid_cls_acc": valid_cls_acc,
                "valid_loss": valid_loss,
            }
        )

    # retrieval validation
    valid_ret_acc, eer = valid_epoch_ver(model, pair_dataloader, DEVICE, config)
    print("Val Ret. Acc {:.04f}%".format(valid_ret_acc))
    metrics.update({"valid_ret_acc": valid_ret_acc, "eer": eer})

    # save model
    save_model(
        model,
        optimizer,
        scheduler,
        metrics,
        epoch,
        os.path.join(config["checkpoint_dir"], "last.pth"),
    )
    print("Saved epoch model")

    # save best model
    if eval_cls:
        if valid_cls_acc >= best_valid_cls_acc:
            best_valid_cls_acc = valid_cls_acc
            save_model(
                model,
                optimizer,
                scheduler,
                metrics,
                epoch,
                os.path.join(config["checkpoint_dir"], "best_cls.pth"),
            )
            # wandb.save(os.path.join(config['checkpoint_dir'], 'best_cls.pth'))
            print("Saved best classification model")

    if valid_ret_acc >= best_valid_ret_acc:
        best_valid_ret_acc = valid_ret_acc
        save_model(
            model,
            optimizer,
            scheduler,
            metrics,
            epoch,
            os.path.join(config["checkpoint_dir"], "best_ret.pth"),
        )
        # wandb.save(os.path.join(config['checkpoint_dir'], 'best_ret.pth'))
        print("Saved best retrieval model")

    # log to tracker
    # if run is not None:
    #     run.log(metrics)


# # Testing and Kaggle Submission (Verification)
def test_epoch_ver(model, pair_data_loader, config):

    model.eval()
    scores = []
    batch_bar = tqdm(
        total=len(pair_data_loader),
        dynamic_ncols=True,
        position=0,
        leave=False,
        desc="Val Veri.",
    )
    for i, (images1, images2) in enumerate(pair_data_loader):

        images = torch.cat([images1, images2], dim=0).to(DEVICE)
        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)

        feats = F.normalize(outputs["feats"], dim=1)
        feats1, feats2 = feats.chunk(2)
        similarity = F.cosine_similarity(feats1, feats2)
        scores.extend(similarity.cpu().numpy().tolist())
        batch_bar.update()

    return scores


def test_all_versions():
    global model, optimizer, scheduler
    print("Testing all versions")

    # load best_cls model and test it
    model, _, _, _, _ = load_model(
        model,
        optimizer,
        scheduler,
        os.path.join(config["checkpoint_dir"], "best_cls.pth"),
    )

    best_cls_scores = test_epoch_ver(model, test_pair_dataloader, config)

    with open("best_cls_submission.csv", "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(best_cls_scores)):
            f.write("{},{}\n".format(i, best_cls_scores[i]))

    # load best_ret model and test it
    model, _, _, _, _ = load_model(
        model,
        optimizer,
        scheduler,
        os.path.join(config["checkpoint_dir"], "best_ret.pth"),
    )

    best_ret_scores = test_epoch_ver(model, test_pair_dataloader, config)

    with open("best_ret_submission.csv", "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(best_ret_scores)):
            f.write("{},{}\n".format(i, best_ret_scores[i]))

    # load last model and test it
    model, _, _, _, _ = load_model(
        model, optimizer, scheduler, os.path.join(config["checkpoint_dir"], "last.pth")
    )

    last_scores = test_epoch_ver(model, test_pair_dataloader, config)

    with open("last_submission.csv", "w+") as f:
        f.write("ID,Label\n")
        for i in range(len(last_scores)):
            f.write("{},{}\n".format(i, last_scores[i]))


test_all_versions()
