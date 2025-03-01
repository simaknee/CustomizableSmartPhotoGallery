from dataset import MiniImagenetDataset
from network import SiameseNetwork
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryConfusionMatrix
from tqdm import tqdm
import numpy as np
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add and parse arguments
    parser.add_argument('--val_path', type=str, help="path for validation dataset", required=True)
    parser.add_argument('--checkpoint_file', type=str, help="path for checkpoint file to load", default='./checkpoints/best.pth')
    parser.add_argument('--device', type=str, help="use 'cuda' as a default, or 'cpu'", default='cuda')
    args = parser.parse_args()

    # load dataset
    val_dataset = MiniImagenetDataset(args.val_path, k=1, n=5)
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # load model
    model = SiameseNetwork()
    checkpoint = torch.load(args.checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # loss metric
    criterion = torch.nn.BCEWithLogitsLoss()

    use_gpu = False
    # if gpu is available and set to use, use cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        model.cuda()
        use_gpu = True

    losses = []
    y_preds = []
    y_true = []

    # evalute loss
    with torch.no_grad():
        for x1, x2, y in tqdm(val_dataloader):
            if use_gpu:
                x1 = x1.to(args.device)
                x2 = x2.to(args.device)
                y = y.to(args.device)
            y_pred = model(x1, x2)
            loss = criterion(y_pred, y.unsqueeze(1).float())
            losses.append(loss.item())

            y_preds.append(y_pred.squeeze(1).detach().cpu())
            y_true.append(y.detach().cpu())

    val_loss = sum(losses)/len(losses)

    # find the threshold that maximize performance (based on F1 score)
    threshold = 0
    max_score = 0
    y_preds = torch.cat(y_preds)
    y_true = torch.cat(y_true)
    for t in np.linspace(0, 1, num=51):
        confusion_matrix = BinaryConfusionMatrix(threshold=t, device='cpu')
        confusion_matrix.update(y_preds, y_true)
        (TN, FP), (FN, TP) = confusion_matrix.compute()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        if max_score < F1:
            max_score = F1
            threshold = t

    # print evaluation results
    print(f"validation loss: {val_loss}")
    print(f"best threshold value: {threshold}")

    confusion_matrix = BinaryConfusionMatrix(threshold=0.1, device='cpu')
    confusion_matrix.update(y_preds, y_true)
    (TN, FP), (FN, TP) = confusion_matrix.compute()

    print(f"\
        TP | FP   {TP.item()} | {FP.item()}\n\
        ------- = ----------\n\
        FN | TN   {FN.item()} | {TN.item()}\
          ")
