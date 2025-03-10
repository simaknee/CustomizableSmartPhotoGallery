from dataset import MiniImagenetDataset
from network import SiameseNetwork
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import torch
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add and parse arguments
    parser.add_argument('--train_path', type=str, help="path for training dataset", required=True)
    parser.add_argument('--val_path', type=str, help="path for validation dataset", required=True)
    parser.add_argument('--checkpoint_path', type=str, help="path for checkpoint files", default='./checkpoints')
    parser.add_argument('--log_path', type=str, help="path for logging files", default='./logs')
    parser.add_argument('--device', type=str, help="use 'cuda' as a default, or 'cpu'", default='cuda')
    parser.add_argument('--backbone', type=str, help="backbone model for simaese network", default='resnet18')
    parser.add_argument('--epoch', type=int, help="the number of training epochs", default=1000)
    args = parser.parse_args()

    # load trainining and validation dataset
    train_dataset = MiniImagenetDataset(args.train_path, k=1, n=5)
    val_dataset = MiniImagenetDataset(args.val_path, k=1, n=5)

    train_dataloader = DataLoader(train_dataset, batch_size=16, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    # load model
    model = SiameseNetwork()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    schedular = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer,
                                                          lr_lambda=lambda epoch: 0.99 ** epoch)
    criterion = torch.nn.BCEWithLogitsLoss()

    writer = SummaryWriter(args.log_path)

    use_gpu = False
    # if gpu is available and set to use, use cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        model.cuda()
        use_gpu = True

    best_loss = 9999999999

    # do train
    for epoch in tqdm(range(args.epoch), desc="training epochs"):
        model.train()

        losses = []

        # training
        for x1, x2, y in train_dataloader:
            if use_gpu:
                x1 = x1.to(args.device)
                x2 = x2.to(args.device)
                y = y.to(args.device)
            y_pred = model(x1, x2)
            loss = criterion(y_pred, y.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = sum(losses)/len(losses)

        writer.add_scalar('train_loss', train_loss, epoch)

        schedular.step()

        model.eval()

        losses = []

        # validation
        for x1, x2, y in val_dataloader:
            if use_gpu:
                x1 = x1.to(args.device)
                x2 = x2.to(args.device)
                y = y.to(args.device)
            y_pred = model(x1, x2)
            loss = criterion(y_pred, y.unsqueeze(1).float())

            losses.append(loss.item())

        val_loss = sum(losses)/len(losses)

        writer.add_scalar('val_loss', val_loss, epoch)

        print(f"train loss: {train_loss}, val loss: {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.checkpoint_path, "best.pth")
            )

        # Save model based on the frequency defined by "args.save_after"
        if (epoch + 1) % 100 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.checkpoint_path, "epoch_{}.pth".format(epoch + 1))
            )
