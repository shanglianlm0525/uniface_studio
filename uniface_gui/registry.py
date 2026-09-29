"""Declarative task and model registry used by the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from uniface import (
    SCRFD,
    AdaFace,
    AgeGender,
    ArcFace,
    BiSeNet,
    BlazeFace,
    CenterFace,
    EdgeFace,
    EDifFIQA,
    Emotion,
    FaceAttribNet,
    FaceMesh,
    FairFace,
    HeadPose,
    Landmark106,
    MiniFASNet,
    MobileFace,
    MobileGaze,
    MODNet,
    PIPNet,
    RetinaFace,
    SphereFace,
    XSeg,
    YOLOv5Face,
    YOLOv8Face,
)
from uniface.constants import (
    AdaFaceWeights,
    AgeGenderWeights,
    ArcFaceWeights,
    BlazeFaceWeights,
    CenterFaceWeights,
    EdgeFaceWeights,
    EDifFIQAWeights,
    EmotionWeights,
    FaceAttribNetWeights,
    FaceMeshWeights,
    FairFaceWeights,
    GazeWeights,
    HeadPoseWeights,
    LandmarkWeights,
    MiniFASNetWeights,
    MobileFaceWeights,
    MODNetWeights,
    ParsingWeights,
    PIPNetWeights,
    RetinaFaceWeights,
    SCRFDWeights,
    SphereFaceWeights,
    XSegWeights,
    YOLOv5FaceWeights,
    YOLOv8FaceWeights,
)


@dataclass(frozen=True)
class ModelSpec:
    """A model class paired with one concrete weight variant."""

    label: str
    cls: type[Any]
    weight: Enum
    supports_providers: bool = True


def _variants(label: str, cls: type[Any], weights: type[Enum], *, providers: bool = True) -> list[ModelSpec]:
    return [ModelSpec(f'{label} / {item.name}', cls, item, providers) for item in weights]


DETECTION_MODELS = [
    *_variants('RetinaFace', RetinaFace, RetinaFaceWeights),
    *_variants('SCRFD', SCRFD, SCRFDWeights),
    *_variants('CenterFace', CenterFace, CenterFaceWeights),
    *_variants('YOLOv5-Face', YOLOv5Face, YOLOv5FaceWeights),
    *_variants('YOLOv8-Face', YOLOv8Face, YOLOv8FaceWeights),
    *_variants('BlazeFace', BlazeFace, BlazeFaceWeights),
]
RECOGNITION_MODELS = [
    *_variants('AdaFace', AdaFace, AdaFaceWeights),
    *_variants('ArcFace', ArcFace, ArcFaceWeights),
    *_variants('EdgeFace', EdgeFace, EdgeFaceWeights),
    *_variants('MobileFace', MobileFace, MobileFaceWeights),
    *_variants('SphereFace', SphereFace, SphereFaceWeights),
]
LANDMARK_MODELS = [
    *_variants('2d106det', Landmark106, LandmarkWeights),
    *_variants('PIPNet', PIPNet, PIPNetWeights),
    *_variants('Face Mesh', FaceMesh, FaceMeshWeights),
]
DEMOGRAPHY_MODELS = [
    *_variants('AgeGender', AgeGender, AgeGenderWeights),
    *_variants('FairFace', FairFace, FairFaceWeights),
]
EMOTION_MODELS = _variants('Emotion', Emotion, EmotionWeights, providers=False)
STATE_MODELS = _variants('FaceAttribNet', FaceAttribNet, FaceAttribNetWeights)
GAZE_MODELS = _variants('MobileGaze', MobileGaze, GazeWeights)
HEADPOSE_MODELS = _variants('HeadPose', HeadPose, HeadPoseWeights)
PARSING_MODELS = [*_variants('BiSeNet', BiSeNet, ParsingWeights), *_variants('XSeg', XSeg, XSegWeights)]
MATTING_MODELS = _variants('MODNet', MODNet, MODNetWeights)
SPOOFING_MODELS = _variants('MiniFASNet', MiniFASNet, MiniFASNetWeights)
QUALITY_MODELS = _variants('eDifFIQA', EDifFIQA, EDifFIQAWeights)

TASK_MODELS: dict[str, list[ModelSpec]] = {
    '人脸检测': DETECTION_MODELS,
    '人脸识别': RECOGNITION_MODELS,
    '面部关键点': LANDMARK_MODELS,
    '人脸解析': PARSING_MODELS,
    '人像抠图': MATTING_MODELS,
    '视线估计': GAZE_MODELS,
    '头部姿态': HEADPOSE_MODELS,
    '人口属性': DEMOGRAPHY_MODELS,
    '表情识别': EMOTION_MODELS,
    '面部状态': STATE_MODELS,
    '人脸质量': QUALITY_MODELS,
    '活体检测': SPOOFING_MODELS,
}

MODEL_BY_LABEL = {model.label: model for models in TASK_MODELS.values() for model in models}


def labels(task: str) -> list[str]:
    """Return model labels for a task."""
    return [model.label for model in TASK_MODELS[task]]


def default_detector_label() -> str:
    """Return the recommended detector model label."""
    return next(model.label for model in DETECTION_MODELS if model.weight == RetinaFaceWeights.MNET_V2)
