"""Inference services shared by the Gradio event handlers."""

# ruff: noqa: RUF001

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import lru_cache
import json
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from uniface import FAISS, BlurFace, BYTETracker, compute_similarity, download_models
from uniface.common import xyxy_to_cxcywh
from uniface.draw import draw_gaze, draw_head_pose, draw_mesh, vis_parsing_maps

from .registry import MODEL_BY_LABEL

OUTPUT_DIR = Path(tempfile.gettempdir()) / 'uniface_gui'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _providers(provider: str) -> list[str] | None:
    if provider == 'CPU':
        return ['CPUExecutionProvider']
    if provider == 'CUDA':
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return None


@lru_cache(maxsize=48)
def get_model(label: str, provider: str = 'Auto') -> Any:
    """Construct and cache a model selected in the GUI."""
    spec = MODEL_BY_LABEL[label]
    kwargs: dict[str, Any] = {'model_name': spec.weight}
    if spec.supports_providers:
        kwargs['providers'] = _providers(provider)
    return spec.cls(**kwargs)


@lru_cache(maxsize=12)
def get_detector(label: str, provider: str, confidence: float) -> Any:
    """Construct and cache a detector with the requested threshold."""
    spec = MODEL_BY_LABEL[label]
    return spec.cls(
        model_name=spec.weight,
        confidence_threshold=round(float(confidence), 3),
        providers=_providers(provider),
    )


def _bgr(image: np.ndarray | None) -> np.ndarray:
    if image is None:
        raise ValueError('请先上传图片或使用摄像头拍摄。')
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _face_row(face: Any, index: int) -> dict[str, Any]:
    return {
        'face': index,
        'bbox': np.asarray(face.bbox).round(2).tolist(),
        'confidence': round(float(face.confidence), 4),
    }


def _label(image: np.ndarray, bbox: np.ndarray, text: str, color: tuple[int, int, int] = (60, 220, 80)) -> None:
    x1, y1, x2, y2 = np.asarray(bbox[:4], dtype=int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    y = max(20, y1 - 8)
    cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _crop(image: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(bbox[:4], dtype=int)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def run_image_task(
    image: np.ndarray | None,
    task: str,
    model_label: str,
    detector_label: str,
    provider: str,
    confidence: float,
    blur_method: str,
) -> tuple[np.ndarray, str]:
    """Run any single-image task and return visualization plus JSON results."""
    source = _bgr(image)
    output = source.copy()
    detector = get_detector(detector_label, provider, confidence)
    faces = detector.detect(source)
    rows: list[dict[str, Any]] = []

    if task == '人脸检测':
        for index, face in enumerate(faces, 1):
            _label(output, face.bbox, f'Face {index} {face.confidence:.2f}')
            row = _face_row(face, index)
            row['landmarks'] = _jsonable(face.landmarks)
            rows.append(row)

    elif task == '人像抠图':
        model = get_model(model_label, provider)
        alpha = np.asarray(model.predict(source), dtype=np.float32)
        if alpha.ndim == 3:
            alpha = np.squeeze(alpha)
        if alpha.shape != source.shape[:2]:
            alpha = cv2.resize(alpha, (source.shape[1], source.shape[0]))
        alpha = np.clip(alpha, 0, 1)[..., None]
        background = np.full_like(source, 245)
        output = (source * alpha + background * (1 - alpha)).astype(np.uint8)
        rows.append({'model': model_label, 'alpha_min': float(alpha.min()), 'alpha_max': float(alpha.max())})

    elif task == '匿名化':
        blurrer = BlurFace(method=blur_method)
        output = blurrer.anonymize(source, faces)
        rows.append({'faces': len(faces), 'method': blur_method})

    else:
        model = get_model(model_label, provider)
        for index, face in enumerate(faces, 1):
            row = _face_row(face, index)
            crop = _crop(source, face.bbox)
            if task == '面部关键点':
                if model.__class__.__name__ == 'FaceMesh':
                    result = model.predict(source, [face])[0]
                    points = result.landmarks
                    draw_mesh(output, points, mode='points')
                    row.update({'points': len(points), 'score': float(result.score)})
                else:
                    points = model.get_landmarks(source, face.bbox)
                    for x, y in np.asarray(points[:, :2], dtype=int):
                        cv2.circle(output, (x, y), 1, (40, 230, 70), -1)
                    row['points'] = len(points)
            elif task == '人脸解析':
                if crop is None:
                    continue
                mask = model.parse(crop)
                if model.__class__.__name__ == 'BiSeNet':
                    visual = vis_parsing_maps(crop, mask, save_image=False)
                else:
                    mask = np.asarray(mask, dtype=np.float32)
                    overlay = np.zeros_like(crop)
                    overlay[:, :, 1] = 255
                    visual = np.where((mask > 0.5)[..., None], cv2.addWeighted(crop, 0.45, overlay, 0.55, 0), crop)
                x1, y1, x2, y2 = np.asarray(face.bbox[:4], dtype=int)
                x1, x2 = np.clip([x1, x2], 0, source.shape[1])
                y1, y2 = np.clip([y1, y2], 0, source.shape[0])
                output[y1:y2, x1:x2] = cv2.resize(visual, (x2 - x1, y2 - y1))
                row['mask_shape'] = list(mask.shape)
            elif task == '视线估计':
                if crop is None:
                    continue
                result = model.estimate(crop)
                draw_gaze(output, face.bbox, result.pitch, result.yaw, draw_angles=True)
                row.update(
                    {'pitch_degrees': float(np.degrees(result.pitch)), 'yaw_degrees': float(np.degrees(result.yaw))}
                )
            elif task == '头部姿态':
                if crop is None:
                    continue
                result = model.estimate(crop)
                draw_head_pose(output, face.bbox, result.pitch, result.yaw, result.roll, draw_type='cube')
                row.update(_jsonable(result))
            elif task in {'人口属性', '表情识别', '面部状态'}:
                result = model.predict(source, face)
                values = _jsonable(result)
                row.update(values)
                summary = ', '.join(f'{key}={value}' for key, value in values.items() if value is not None)
                _label(output, face.bbox, summary[:90])
            elif task == '人脸质量':
                result = model.predict(source, face.landmarks)
                row.update(_jsonable(result))
                _label(output, face.bbox, f'Quality {result.score:.3f}')
            elif task == '活体检测':
                result = model.predict(source, face.bbox)
                row.update(_jsonable(result))
                color = (50, 220, 80) if result.is_real else (30, 30, 230)
                _label(output, face.bbox, f'{"Real" if result.is_real else "Spoof"} {result.confidence:.3f}', color)
            rows.append(row)

    payload = {
        'task': task,
        'model': model_label if task != '匿名化' else blur_method,
        'faces': len(faces),
        'results': rows,
    }
    return _rgb(output), json.dumps(_jsonable(payload), ensure_ascii=False, indent=2)


def compare_faces(
    image_a: np.ndarray | None,
    image_b: np.ndarray | None,
    model_label: str,
    detector_label: str,
    provider: str,
    threshold: float,
) -> tuple[str, str]:
    """Compare the first detected face in two images."""
    first, second = _bgr(image_a), _bgr(image_b)
    detector = get_detector(detector_label, provider, 0.5)
    faces_a, faces_b = detector.detect(first), detector.detect(second)
    if not faces_a or not faces_b:
        raise ValueError('一张或两张图片中未检测到人脸。')
    if getattr(detector, 'supports_alignment', True) is False:
        raise ValueError('BlazeFace 不提供识别所需的 5 点对齐关键点，请换用其他检测器。')
    recognizer = get_model(model_label, provider)
    emb_a = recognizer.get_normalized_embedding(first, faces_a[0].landmarks)
    emb_b = recognizer.get_normalized_embedding(second, faces_b[0].landmarks)
    similarity = float(compute_similarity(emb_a, emb_b, normalized=True))
    matched = similarity >= threshold
    return (
        f'{"同一人" if matched else "不同人"} · 相似度 {similarity:.4f}',
        json.dumps(
            {'similarity': similarity, 'threshold': threshold, 'matched': matched}, ensure_ascii=False, indent=2
        ),
    )


def _assign_track_ids(faces: list[Any], tracks: np.ndarray) -> list[Any]:
    if not len(tracks) or not faces:
        return []
    face_centers = xyxy_to_cxcywh(np.asarray([face.bbox for face in faces], dtype=np.float32))[:, :2]
    track_centers = xyxy_to_cxcywh(tracks[:, :4])[:, :2]
    for index, track in enumerate(tracks):
        distances = np.square(track_centers[index] - face_centers).sum(axis=1)
        faces[int(np.argmin(distances))].track_id = int(track[4])
    return [face for face in faces if face.track_id is not None]


def run_video_task(
    video_path: str | None,
    operation: str,
    detector_label: str,
    provider: str,
    confidence: float,
    blur_method: str,
) -> tuple[str, str]:
    """Run detection, tracking, or anonymization over an uploaded video."""
    if not video_path:
        raise ValueError('请先上传视频。')
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError('无法打开上传的视频。')
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = OUTPUT_DIR / f'{uuid4().hex}.mp4'
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    detector = get_detector(detector_label, provider, confidence)
    tracker = BYTETracker(track_thresh=confidence) if operation == '人脸跟踪' else None
    blurrer = BlurFace(method=blur_method) if operation == '匿名化' else None
    frame_count = detection_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_count += 1
        faces = detector.detect(frame)
        detection_count += len(faces)
        if tracker is not None:
            detections = np.asarray([[*face.bbox, face.confidence] for face in faces], dtype=np.float32)
            if not len(detections):
                detections = np.empty((0, 5), dtype=np.float32)
            tracks = tracker.update(detections)
            for face in _assign_track_ids(faces, tracks):
                _label(frame, face.bbox, f'ID {face.track_id}')
        elif blurrer is not None:
            frame = blurrer.anonymize(frame, faces, inplace=True)
        else:
            for face in faces:
                _label(frame, face.bbox, f'{face.confidence:.2f}')
        writer.write(frame)
    capture.release()
    writer.release()
    summary = {'operation': operation, 'frames': frame_count, 'detections': detection_count, 'output': str(output_path)}
    return str(output_path), json.dumps(summary, ensure_ascii=False, indent=2)


def build_vector_index(
    files: list[str] | None,
    labels_text: str,
    db_path: str,
    recognition_label: str,
    detector_label: str,
    provider: str,
) -> str:
    """Create a FAISS index from uploaded face images."""
    if not files:
        raise ValueError('请上传至少一张注册图片。')
    names = [line.strip() for line in labels_text.splitlines() if line.strip()]
    if len(names) != len(files):
        raise ValueError('身份标签必须每行一个，并与上传图片数量一致。')
    detector = get_detector(detector_label, provider, 0.5)
    recognizer = get_model(recognition_label, provider)
    store = FAISS(db_path=db_path)
    added = 0
    for path, name in zip(files, names, strict=True):
        image = cv2.imread(path)
        faces = detector.detect(image) if image is not None else []
        if not faces:
            continue
        embedding = recognizer.get_normalized_embedding(image, faces[0].landmarks)
        store.add(embedding, {'person_id': name, 'source': path})
        added += 1
    if not added:
        raise ValueError('所有注册图片均未检测到有效人脸。')
    store.save()
    return f'索引已保存：{db_path}（成功添加 {added}/{len(files)} 张）'


def search_vector_index(
    image: np.ndarray | None,
    db_path: str,
    recognition_label: str,
    detector_label: str,
    provider: str,
    threshold: float,
) -> tuple[np.ndarray, str]:
    """Search every face in an image against a persisted FAISS index."""
    source = _bgr(image)
    output = source.copy()
    store = FAISS(db_path=db_path)
    if not store.load():
        raise ValueError(f'未找到索引：{db_path}')
    detector = get_detector(detector_label, provider, 0.5)
    recognizer = get_model(recognition_label, provider)
    results = []
    for index, face in enumerate(detector.detect(source), 1):
        embedding = recognizer.get_normalized_embedding(source, face.landmarks)
        match, similarity = store.search(embedding, threshold=threshold)
        name = match.get('person_id', 'Unknown') if match else 'Unknown'
        _label(output, face.bbox, f'{name} {similarity:.3f}')
        results.append({'face': index, 'person_id': name, 'similarity': float(similarity)})
    return _rgb(output), json.dumps(results, ensure_ascii=False, indent=2)


def download_selected_models(labels: list[str] | None) -> str:
    """Download and verify selected model weights."""
    if not labels:
        raise ValueError('请至少选择一个模型。')
    weights = [MODEL_BY_LABEL[label].weight for label in labels]
    paths = download_models(weights)
    lines = [f'✓ {weight.value}: {path}' for weight, path in paths.items()]
    return '\n'.join(lines)
