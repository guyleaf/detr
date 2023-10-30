"""
Plotting utilities to visualize training logs.
"""
import warnings
from pathlib import Path, PurePath
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from util.misc import interpolate

COCO_EVAL_STAT_COLUMNS = [
    "mAP",
    "mAP@.50",
    "mAP@.75",
    "mAP@s",
    "mAP@m",
    "mAP@l",
    "mAR@1",
    "mAR@10",
    "mAR@100",
    "mAR@s",
    "mAR@m",
    "mAR@l",
]


def convert_overlay_heatmap(
    feat_map: Union[np.ndarray, torch.Tensor],
    img: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3 and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img


def img_from_canvas(canvas: FigureCanvasAgg) -> np.ndarray:
    """Get RGB image from ``FigureCanvasAgg``.

    Args:
        canvas (FigureCanvasAgg): The canvas to get image.

    Returns:
        np.ndarray: the output of image in RGB.
    """  # noqa: E501
    s, (width, height) = canvas.print_to_buffer()
    buffer = np.frombuffer(s, dtype="uint8")
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    return rgb.astype("uint8")


def auto_arrange_images(
    image_list: list, image_column: int = 2, padding: int = 5, padding_value: int = 255
) -> np.ndarray:
    """Auto arrange image to image_column x N row.
    Refers to https://github.com/open-mmlab/mmyolo/blob/8c4d9dc503dc8e327bec8147e8dc97124052f693/mmyolo/utils/misc.py#L25

    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column. Default: 2.
    Return:
        (np.ndarray): image_column x N row merge image
    """
    img_count = len(image_list)
    if img_count <= image_column:
        # no need to arrange
        image_show = np.concatenate(image_list, axis=1)
    else:
        # arrange image according to image_column
        image_row = round(img_count / image_column)
        image_shape = list(image_list[0].shape)
        height, width = image_shape[0] + padding, image_shape[1] + padding
        image_shape[0] += height * (image_row - 1)
        image_shape[1] += width * (image_column - 1)
        image_show = np.full(image_shape, padding_value, dtype=np.uint8)

        for i in range(image_row):
            start_col = image_column * i
            start_h = i * height
            end_h = start_h + height - padding
            for j in range(image_column):
                start_w = j * width
                end_w = start_w + width - padding
                image_show[start_h:end_h, start_w:end_w] = image_list[start_col + j]

    return image_show


def plot_logs(
    logs,
    fields=("class_error", "loss_bbox_unscaled", "mAP"),
    ewm_col=0,
    log_name="log.txt",
):
    """
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    """
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(
                f"{func_name} info: logs param expects a list argument, converted to list[Path]."
            )
        else:
            raise ValueError(
                f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}"
            )

    # Quality checks - verify valid dir(s), that every item in list is Path object, and that log_name exists in each dir
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(
                f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}"
            )
        if not dir.exists():
            raise ValueError(
                f"{func_name} - invalid directory in logs argument:\n{dir}"
            )
        # verify log_name exists
        fn = Path(dir / log_name)
        if not fn.exists():
            print(f"-> missing {log_name}.  Have you gotten to Epoch 1 in training?")
            print(f"--> full path of missing log file: {fn}")
            return

    # load log file(s) and plot
    dfs: list[pd.DataFrame] = [
        pd.read_json(Path(p) / log_name, lines=True) for p in logs
    ]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(32, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        coco_eval = pd.DataFrame(
            np.stack(df.test_coco_eval_bbox.dropna().values),
            columns=COCO_EVAL_STAT_COLUMNS,
        )

        for j, field in enumerate(fields):
            if field in coco_eval:
                coco_metric = coco_eval[field].ewm(com=ewm_col).mean()
                axs[j].plot(coco_metric, c=color)
            else:
                df.select_dtypes(include="number").interpolate().ewm(
                    com=ewm_col
                ).mean().plot(
                    y=[f"train_{field}", f"test_{field}"],
                    ax=axs[j],
                    color=[color] * 2,
                    style=["-", "--"],
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)
    return fig, axs


def plot_precision_recall(files, naming_scheme="iter"):
    if naming_scheme == "exp_id":
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == "iter":
        names = [f.stem for f in files]
    else:
        raise ValueError(f"not supported {naming_scheme}")
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(
        files, sns.color_palette("Blues", n_colors=len(files)), names
    ):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data["precision"]
        recall = data["params"].recThrs
        scores = data["scores"]
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data["recall"][0, :, 0, -1].mean()
        print(
            f"{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, "
            + f"score={scores.mean():0.3f}, "
            + f"f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}"
        )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title("Precision / Recall")
    axs[0].legend(names)
    axs[1].set_title("Scores / Recall")
    axs[1].legend(names)
    return fig, axs


def plot_image(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    line_thickness: int = 2,
    palette: Optional[str] = "cool",
    font_face: int = cv2.FONT_HERSHEY_COMPLEX,
    font_scale: float = 0.5,
    font_thickness: int = 1,
):
    cmap = plt.colormaps.get_cmap(palette)
    boxes = np.rint(boxes).astype(int)

    for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
        color = tuple(map(int, cmap(label, bytes=True)[:-1]))[::-1]

        # draw box
        image = cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=line_thickness,
        )

        # draw text on image
        text = classes[label]
        text = f"{text}: {score:.2f}"
        (text_width, text_height), base_line = cv2.getTextSize(
            text, font_face, font_scale, font_thickness
        )
        text_top_left = (x1, y1 - text_height - base_line)
        text_bottom_right = (x1 + text_width, y1)

        # text background
        image = cv2.rectangle(
            image, text_top_left, text_bottom_right, color, thickness=-1
        )
        # text
        image = cv2.putText(
            image,
            text,
            (x1, y1 - base_line),
            font_face,
            font_scale,
            (0, 0, 0),
            thickness=font_thickness,
        )

    return image


def plot_featmap(
    featmap: torch.Tensor,
    overlaid_image: Optional[np.ndarray] = None,
    channel_reduction: Optional[str] = "squeeze_mean",
    topk: int = 20,
    arrangement: Tuple[int, int] = (4, 5),
    resize_shape: Optional[tuple] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw featmap.
    Copied from https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md#feature-map-visualization.

    - If `overlaid_image` is not None, the final output image will be the
      weighted sum of img and featmap.

    - If `resize_shape` is specified, `featmap` and `overlaid_image`
      are interpolated.

    - If `resize_shape` is None and `overlaid_image` is not None,
      the feature map will be interpolated to the spatial size of the image
      in the case where the spatial dimensions of `overlaid_image` and
      `featmap` are different.

    - If `channel_reduction` is "squeeze_mean" and "select_max",
      it will compress featmap to single channel image and weighted
      sum to `overlaid_image`.

    - If `channel_reduction` is None

      - If topk <= 0, featmap is assert to be one or three
        channel and treated as image and will be weighted sum
        to ``overlaid_image``.
      - If topk > 0, it will select topk channel to show by the sum of
        each channel. At the same time, you can specify the `arrangement`
        to set the window layout.

    Args:
        featmap (torch.Tensor): The featmap to draw which format is
            (C, H, W).
        overlaid_image (np.ndarray, optional): The overlaid image.
            Defaults to None.
        channel_reduction (str, optional): Reduce multiple channels to a
            single channel. The optional value is 'squeeze_mean'
            or 'select_max'. Defaults to 'squeeze_mean'.
        topk (int): If channel_reduction is not None and topk > 0,
            it will select topk channel to show by the sum of each channel.
            if topk <= 0, tensor_chw is assert to be one or three.
            Defaults to 20.
        arrangement (Tuple[int, int]): The arrangement of featmap when
            channel_reduction is not None and topk > 0. Defaults to (4, 5).
        resize_shape (tuple, optional): The shape to scale the feature map.
            Defaults to None.
        alpha (Union[int, List[int]]): The transparency of featmap.
            Defaults to 0.5.

    Returns:
        np.ndarray: RGB image.
    """
    import matplotlib.pyplot as plt

    assert isinstance(featmap, torch.Tensor), (
        f"`featmap` should be torch.Tensor," f" but got {type(featmap)}"
    )
    assert featmap.ndim == 3, f"Input dimension must be 3, " f"but got {featmap.ndim}"
    featmap = featmap.detach().cpu()

    if overlaid_image is not None:
        if overlaid_image.ndim == 2:
            overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_GRAY2RGB)

        if overlaid_image.shape[:2] != featmap.shape[1:]:
            warnings.warn(
                f"Since the spatial dimensions of "
                f"overlaid_image: {overlaid_image.shape[:2]} and "
                f"featmap: {featmap.shape[1:]} are not same, "
                f"the feature map will be interpolated. "
                f"This may cause mismatch problems ！"
            )
            if resize_shape is None:
                featmap = interpolate(
                    featmap[None],
                    overlaid_image.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )[0]

    if resize_shape is not None:
        featmap = interpolate(
            featmap[None], resize_shape, mode="bilinear", align_corners=False
        )[0]
        if overlaid_image is not None:
            overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

    if channel_reduction is not None:
        assert channel_reduction in ["squeeze_mean", "select_max"], (
            f'Mode only support "squeeze_mean", "select_max", '
            f"but got {channel_reduction}"
        )
        if channel_reduction == "select_max":
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, 1)
            feat_map = featmap[indices]
        else:
            feat_map = torch.mean(featmap, dim=0)
        return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
    elif topk <= 0:
        featmap_channel = featmap.shape[0]
        assert featmap_channel in [1, 3], (
            "The input tensor channel dimension must be 1 or 3 "
            "when topk is less than 1, but the channel "
            f"dimension you input is {featmap_channel}, you can use the"
            " channel_reduction parameter or set topk greater than "
            "0 to solve the error"
        )
        return convert_overlay_heatmap(featmap, overlaid_image, alpha)
    else:
        row, col = arrangement
        channel, height, width = featmap.shape
        assert row * col >= topk, (
            "The product of row and col in "
            "the `arrangement` is less than "
            "topk, please set the "
            "`arrangement` correctly"
        )

        # Extract the feature map of topk
        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
        _, indices = torch.topk(sum_channel_featmap, topk)
        topk_featmap = featmap[indices]

        fig = plt.figure(frameon=False)
        # Set the window layout
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        dpi = fig.get_dpi()
        fig.set_size_inches((width * col + 1e-2) / dpi, (height * row + 1e-2) / dpi)
        for i in range(topk):
            axes = fig.add_subplot(row, col, i + 1)
            axes.axis("off")
            axes.text(2, 15, f"channel: {indices[i]}", fontsize=10)
            axes.imshow(convert_overlay_heatmap(topk_featmap[i], overlaid_image, alpha))
        image = img_from_canvas(fig.canvas)
        plt.close(fig)
        return image
