from __future__ import annotations

import os

import torch.nn as nn
from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    """YuJian one-class fish detector based on YOLOX-Nano."""

    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.25
        self.num_classes = 1
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.mixup_prob = 0.0
        self.data_dir = os.environ.get("DETECTOR_DATASET_ROOT", "/tmp/yujian-detector-dataset")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        self.data_num_workers = int(os.environ.get("DETECTOR_DATA_WORKERS", "4"))
        self.max_epoch = int(os.environ.get("DETECTOR_EPOCHS", "30"))
        self.no_aug_epochs = min(5, max(1, self.max_epoch // 5))
        self.warmup_epochs = min(2, max(1, self.max_epoch // 10))
        self.eval_interval = max(1, int(os.environ.get("DETECTOR_EVAL_INTERVAL", "2")))
        self.print_interval = 10
        self.save_history_ckpt = False
        self.output_dir = os.environ.get("DETECTOR_OUTPUT_DIR", "/tmp/yolox_outputs")
        self.exp_name = "yujian_fish_yolox_nano"
        # Evaluation threshold stays low; production threshold is owned by recognition_pipeline_v1.json.
        self.test_conf = 0.01
        self.nmsthre = 0.45
        self.seed = int(os.environ.get("DETECTOR_SEED", "20260831"))

    def get_model(self, sublinear: bool = False):
        def init_yolo(module):
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eps = 1e-3
                    layer.momentum = 0.03

        if getattr(self, "model", None) is None:
            from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=True,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
