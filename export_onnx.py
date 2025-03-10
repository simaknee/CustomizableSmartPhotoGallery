import torch
import argparse
from network import SiameseNetwork


def export_onnx(checkpoint_path: str, export_path: str, backbone='resnet18'):
    """ 
        export trained siamese network model using onnx

        parameters:
            checkpoint_path: path of checkpoint file
            export_path: path to export
            backbone: backbone name of model

    """
    # load model from the checkpoint
    model = SiameseNetwork()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # define dummy input
    dummy_input1 = torch.randn(1, 3, 224, 224)
    dummy_input2 = torch.randn(1, 3, 224, 224)

    # onnx conversion
    torch.onnx.export(
        model,
        (dummy_input1, dummy_input2),
        export_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['img1', 'img2'],
        output_names=['similarity'],
        dynamic_axes={'img1': {0: 'batch_size'}, 'img2': {0: 'batch_size'}, 'similarity': {0: 'batch_size'}}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add and parse arguments
    parser.add_argument('--checkpoint_path', type=str, help="path for model checkpoint file", required=True)
    parser.add_argument('--export_path', type=str, help="path to export onnx file", required=True)
    parser.add_argument('--backbone', type=str, help="backbone name of your model", default='resnet18')
    args = parser.parse_args()

    export_onnx(args.checkpoint_path, args.export_path, args.backbone)
