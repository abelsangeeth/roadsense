"""Class-agnostic traffic-sign detector (Faster R-CNN, ResNet-50-FPN v2, COCO-pretrained).

Recognition is delegated to SignNet: GTSDB has ~600 boxes over 43 classes (a dozen per
class), GTSRB has ~39k crops, so a two-stage design -- "where is a sign" here, "which
sign" in SignNet -- uses each dataset for what it is big enough to teach.
"""
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_detector(pretrained=True, train_sizes=(704, 736, 768, 800, 832, 864, 896), test_size=800):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=train_sizes, max_size=1600,
                                       box_detections_per_img=50)
    # GTSDB signs are 16-128 px wide: shift the anchor pyramid one octave down from the
    # COCO default (32..512). Same 3 ratios per level, so the pretrained RPN head still fits.
    model.rpn.anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                                                 aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)  # background + sign
    model.test_size = test_size
    return model


def build_damage_detector(num_classes=4, pretrained=True, train_sizes=(544, 576, 608, 640, 672, 704), test_size=640):
    """Road-damage detector for dashcam frames (RDD2022): cracks and potholes.

    Damage boxes are large and elongated (a longitudinal crack can span half the frame), so
    the COCO anchor pyramid (32..512) is kept, unlike the sign detector's shifted one.
    """
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=train_sizes, max_size=1024,
                                       box_detections_per_img=50)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes + 1)  # + background
    model.test_size = test_size
    return model


def set_eval_size(model, size):
    model.transform.min_size = (size,)
    return model
