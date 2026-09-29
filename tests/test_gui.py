"""Tests for the declarative GUI model registry and lightweight handlers."""

from unittest.mock import Mock, patch

import numpy as np

from uniface.types import Face
from uniface_gui.engine import run_image_task
from uniface_gui.registry import DETECTION_MODELS, MODEL_BY_LABEL, TASK_MODELS, default_detector_label


def test_registry_exposes_every_model_variant_once():
    labels = [model.label for models in TASK_MODELS.values() for model in models]

    assert len(labels) == len(set(labels))
    assert set(labels) == set(MODEL_BY_LABEL)
    assert len(DETECTION_MODELS) == 16
    assert {'人脸检测', '人脸识别', '面部关键点', '人脸解析', '人像抠图', '视线估计'} <= TASK_MODELS.keys()


def test_default_detector_is_alignment_capable_retinaface():
    assert default_detector_label() == 'RetinaFace / MNET_V2'


def test_detection_handler_returns_rgb_visual_and_json_without_loading_weights():
    detector = Mock()
    detector.detect.return_value = [
        Face(
            bbox=np.array([1, 1, 8, 8], dtype=np.float32),
            confidence=0.9,
            landmarks=np.zeros((5, 2), dtype=np.float32),
        )
    ]
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch('uniface_gui.engine.get_detector', return_value=detector):
        output, payload = run_image_task(
            image,
            '人脸检测',
            'RetinaFace / MNET_025',
            'RetinaFace / MNET_V2',
            'CPU',
            0.5,
            'pixelate',
        )

    assert output.shape == image.shape
    assert '"faces": 1' in payload
    assert '"confidence": 0.9' in payload
