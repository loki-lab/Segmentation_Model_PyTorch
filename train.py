import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.train import TrainEpoch
from segmentation_models_pytorch.utils.train import ValidEpoch
from segmentation_models_pytorch.utils.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torch.optim import Adam
from data import TumorDataset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

model = smp.create_model(arch="unetplusplus",
                         encoder_name="resnet34",
                         encoder_weights="imagenet",
                         in_channels=1,
                         classes=1)

root_dir = "./dataset"

dataset = TumorDataset(root_dir)

train_ds, val_ds = random_split(dataset, [2936, 128])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=12)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

loss = DiceLoss()
metrics = [IoU(threshold=0.5)]

optimizer = Adam([dict(params=model.parameters(), lr=0.0005)]),

train_epoch = TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
)

valid_epoch = ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
)

max_score = 0

for i in range(0, 40):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(val_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')
