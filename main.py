import os

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import click

from models.attribute_predictor_arch import resnet50


# https://github.com/yumingj/Talk-to-Edit
def transform_image(image, resize=False):
    # transform image range to [0, 1]
    if isinstance(image, Image.Image):
        image = TF.to_tensor(image).unsqueeze(0)
    else:
        image = torch.clamp((image + 1) / 2, 0, 1)
    if resize:
        image = F.interpolate(image, (128, 128), mode='area')

    # normalize image to imagenet range
    img_mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(image.device)
    img_std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(image.device)
    image = (image - img_mean) / img_std

    return image


def predictor_to_label(predictor_output):
    scores = []
    labels = []
    for attr_idx in range(len(predictor_output)):
        _, label = torch.max(input=predictor_output[attr_idx], dim=1)
        label = label.cpu().numpy()[0]
        labels.append(label)

        score_per_attr = predictor_output[attr_idx].cpu().numpy()[0]
        # softmax
        score_per_attr = (np.exp(score_per_attr) / np.sum(np.exp(score_per_attr)))[label]
        scores.append(score_per_attr)

    return labels, scores


@click.command()
@click.option("--input-dir", help='The images folder.')
@click.option("--output-file", default="result.txt", help='The result file.')
@torch.no_grad()
def main(input_dir, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict = resnet50(attr_file='configs/talk_resnet50_attributes_5.json')
    ckpt_attr = torch.load('snapshot/predictor_128.pth.tar')
    predict.load_state_dict(ckpt_attr['state_dict'], strict=True)
    predict.eval().to(device)

    files = list(os.listdir(input_dir))
    for step, file_name in enumerate(files):
        image = Image.open(os.path.join(input_dir, file_name))
        image = transform_image(image, resize=True)

        image = image.to(device)
        output = predict(image)
        labels, scores = predictor_to_label(output)

        line = f"{file_name},{str(labels[0])},{str(labels[1])},{str(labels[2])},{str(labels[3])},{str(labels[4])}\n"
        open(output_file, "a+").write(line)

        print(f"Progress: {step + 1}/{len(files)}", end="\r", flush=True)
    print("\nComplete!")


if __name__ == '__main__':
    main()
