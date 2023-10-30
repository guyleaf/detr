import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as T
from torch.utils.hooks import RemovableHandle

from models.detr import DETR, build_model, build_postprocessors
from models.segmentation import DETRsegm
from util.misc import NestedTensor
from util.plot_utils import auto_arrange_images, plot_featmap, plot_image

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def get_arg_parser():
    parser = argparse.ArgumentParser("Visualize attentions")
    # 1st method: read images from image folder or image file
    parser.add_argument(
        "inputs",
        type=str,
        help="Input image file/folder path/coco path.",
    )
    parser.add_argument("weights", type=str, help="Checkpoint file")
    parser.add_argument("--output-dir", type=str, default="outputs")
    # 2nd method: read images from coco annotation
    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help="COCO annotation .json file."
        "The program will try to read images from it.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="The classes of the model.",
    )
    # # 3rd method: use dataset
    # parser.add_argument("--dataset-file", type=str, default=None)
    # parser.add_argument("--image-set", type=str, default="val")
    # parser.add_argument("--")

    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--attention-plot-width",
        type=int,
        default=5,
        help="The width of plot of attention maps",
    )
    parser.add_argument(
        "--pred-score-thr", type=float, default=0.3, help="bbox score threshold"
    )
    # parser.add_argument(
    #     "--batch-size", type=int, default=1, help="Inference batch size."
    # )

    return parser


class AttentionWrapper:
    def __init__(self, model: Union[DETR, DETRsegm]) -> None:
        self.model = model
        self.backbone_features: NestedTensor
        self.decoder_cross_attentions: list[torch.Tensor] = []
        self.handles: list[RemovableHandle] = []

        self.handles.append(
            self.model.backbone[-2].register_forward_hook(self._save_backbone_features)
        )

        if isinstance(self.model, DETRsegm):
            decoder = self.model.detr.transformer.decoder
        else:
            decoder = self.model.transformer.decoder

        for name, module in decoder.named_modules():
            # only visualize cross-attention layer
            if (
                isinstance(module, torch.nn.MultiheadAttention)
                and "multihead_attn" in name
            ):
                self.handles.append(
                    module.register_forward_hook(self._save_decoder_cross_attentions)
                )

    def _save_backbone_features(self, module, input, output):
        self.backbone_features = output["0"]

    def _save_decoder_cross_attentions(self, module, input, output):
        self.decoder_cross_attentions.append(output[1])

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> tuple[Any, NestedTensor, list[torch.Tensor]]:
        self.decoder_cross_attentions = []
        return (
            self.model(*args, **kwargs),
            self.backbone_features,
            self.decoder_cross_attentions,
        )

    def release(self):
        for handle in self.handles:
            handle.remove()


def load_checkpoint(weights: str):
    if weights.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            weights, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(weights, map_location="cpu")
    return checkpoint


@torch.no_grad()
def main(args):
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.weights)

    model_args = checkpoint["args"]
    assert not model_args.masks, "Currently only supports detection task."
    model_args.device = args.device
    model, _ = build_model(model_args)
    model.to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = AttentionWrapper(model)

    transform = T.Compose(
        [
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    postprocessor = build_postprocessors(model_args)["bbox"]

    inputs: list[Path] = args.inputs
    for input in inputs:
        image = Image.open(input).convert("RGB")
        image_size = image.size[::-1]

        transformed_images: torch.Tensor = transform(image).unsqueeze(0)
        transformed_images = transformed_images.to(device)

        outputs, backbone_features, decoder_cross_attentions = model(transformed_images)
        outputs = postprocessor(outputs, torch.tensor([image_size], device=device))

        # filter boxes with pred-score_thr
        output = outputs[0]
        scores = output["scores"].cpu().numpy()
        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()

        positive_indices = scores > args.pred_score_thr
        scores = scores[positive_indices]
        boxes = boxes[positive_indices]
        labels = labels[positive_indices]

        total_positive_bboxes = np.count_nonzero(positive_indices)
        if np.count_nonzero(positive_indices) == 0:
            continue

        # draw bounding boxes on image
        image = np.ascontiguousarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        drawn_image = plot_image(image, boxes, scores, labels, args.classes)

        file_path = output_dir / input.name
        cv2.imwrite(str(file_path), drawn_image)

        # draw attention maps on image
        plot_h, plot_w = (
            math.ceil(total_positive_bboxes / args.attention_plot_width),
            args.attention_plot_width,
        )
        if plot_h == 1:
            plot_w = total_positive_bboxes
        feature_height, feature_width = backbone_features.tensors.shape[-2:]

        drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)
        results = []
        for decoder_cross_attention in decoder_cross_attentions:
            decoder_cross_attention = decoder_cross_attention[0, positive_indices]
            decoder_cross_attention = decoder_cross_attention.view(
                -1, feature_height, feature_width
            )

            result = plot_featmap(
                decoder_cross_attention,
                drawn_image,
                channel_reduction=None,
                topk=total_positive_bboxes,
                arrangement=(plot_h, plot_w),
            )
            results.append(result)
        results = auto_arrange_images(results)
        results = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)

        file_path = output_dir / f"{input.stem}_attention.{input.suffix}"
        cv2.imwrite(str(file_path), results)

    model.release()


def preprocess_inputs(args):
    # extract image paths and captions if it is a COCO annotation file
    annotation = args.annotation
    if annotation is not None:
        base_path = Path(args.inputs)
        inputs = []
        with open(annotation, "r") as f:
            content = json.load(f)
            classes = [
                category["name"]
                for category in sorted(
                    content["categories"], key=lambda category: category["id"]
                )
            ]

            for image in content["images"]:
                file_path = base_path / image["file_name"]
                inputs.append(file_path)
        args.inputs = inputs
        args.classes = classes
    else:
        folder = Path(args.inputs)
        if folder.is_dir():
            args.inputs = list(folder.rglob("|".join(IMG_EXTENSIONS)))
        else:
            args.inputs = [folder]

    assert args.classes is not None
    return args


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    args = preprocess_inputs(args)

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
