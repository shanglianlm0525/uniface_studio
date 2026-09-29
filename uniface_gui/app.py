"""Gradio user interface for all UniFace tasks."""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from .engine import (
    build_vector_index,
    compare_faces,
    download_selected_models,
    run_image_task,
    run_video_task,
    search_vector_index,
)
from .registry import (
    DETECTION_MODELS,
    MODEL_BY_LABEL,
    RECOGNITION_MODELS,
    TASK_MODELS,
    default_detector_label,
    labels,
)

CSS = """
.gradio-container {max-width: 1440px !important; margin: auto !important;}
.hero {padding: 18px 22px; border: 1px solid var(--border-color-primary); border-radius: 16px;
       background: linear-gradient(120deg, rgba(30,136,229,.12), rgba(0,200,160,.08));}
.hero h1 {margin: 0 0 6px 0; font-size: 2rem;}
.hint {color: var(--body-text-color-subdued); font-size: .92rem;}
footer {display: none !important;}
"""

IMAGE_TASKS = [task for task in TASK_MODELS if task != '人脸识别'] + ['匿名化']
DETECTOR_LABELS = [model.label for model in DETECTION_MODELS]
RECOGNIZER_LABELS = [model.label for model in RECOGNITION_MODELS]
ALL_MODEL_LABELS = list(MODEL_BY_LABEL)


def _task_models(task: str):
    choices = labels(task) if task in TASK_MODELS else ['内置模糊算法']
    return gr.Dropdown(choices=choices, value=choices[0], interactive=task != '匿名化')


def _run_image(*args):
    try:
        return run_image_task(*args)
    except Exception as error:
        raise gr.Error(str(error)) from error


def _compare(*args):
    try:
        return compare_faces(*args)
    except Exception as error:
        raise gr.Error(str(error)) from error


def _run_video(*args):
    try:
        return run_video_task(*args)
    except Exception as error:
        raise gr.Error(str(error)) from error


def _build_index(*args):
    try:
        return build_vector_index(*args)
    except Exception as error:
        raise gr.Error(str(error)) from error


def _search_index(*args):
    try:
        return search_vector_index(*args)
    except Exception as error:
        raise gr.Error(str(error)) from error


def _download(*args):
    try:
        return download_selected_models(*args)
    except Exception as error:
        raise gr.Error(str(error)) from error


def build_app() -> gr.Blocks:
    """Build the complete UniFace Gradio application."""
    with gr.Blocks(title='UniFace Studio', css=CSS, theme=gr.themes.Soft()) as demo:
        gr.HTML(
            '<div class="hero"><h1>UniFace Studio</h1>'
            '<div>覆盖检测、识别、跟踪、关键点、解析、抠图、属性、活体、质量、匿名化与向量检索。</div>'
            '<div class="hint">模型权重首次使用时自动下载并校验 SHA-256；较大模型请耐心等待。</div></div>'
        )

        with gr.Accordion('全局运行设置', open=False):
            with gr.Row():
                provider = gr.Dropdown(['Auto', 'CPU', 'CUDA'], value='Auto', label='推理设备')
                detector = gr.Dropdown(DETECTOR_LABELS, value=default_detector_label(), label='公共人脸检测器')
                detection_threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label='检测置信度')
            gr.Markdown('识别、质量等依赖 5 点对齐的任务请勿选择 BlazeFace；界面会在运行时给出提示。')

        with gr.Tabs():
            with gr.Tab('图片实验台'):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label='输入图片',
                            type='numpy',
                            sources=['upload', 'webcam', 'clipboard'],
                        )
                        image_task = gr.Dropdown(IMAGE_TASKS, value='人脸检测', label='Task')
                        image_model = gr.Dropdown(labels('人脸检测'), value=labels('人脸检测')[0], label='Model')
                        blur_method = gr.Dropdown(
                            ['gaussian', 'pixelate', 'blackout', 'elliptical', 'median'],
                            value='pixelate',
                            label='匿名化方式（仅匿名化任务使用）',
                        )
                        image_run = gr.Button('运行图片任务', variant='primary')
                    with gr.Column(scale=2):
                        image_output = gr.Image(label='可视化结果', type='numpy')
                        image_json = gr.Code(label='结构化结果', language='json')
                image_task.change(_task_models, image_task, image_model)
                image_run.click(
                    _run_image,
                    [
                        image_input,
                        image_task,
                        image_model,
                        detector,
                        provider,
                        detection_threshold,
                        blur_method,
                    ],
                    [image_output, image_json],
                )

            with gr.Tab('人脸识别 / 双图比对'):
                with gr.Row():
                    recognition_a = gr.Image(label='图片 A', type='numpy', sources=['upload', 'webcam', 'clipboard'])
                    recognition_b = gr.Image(label='图片 B', type='numpy', sources=['upload', 'webcam', 'clipboard'])
                with gr.Row():
                    recognition_model = gr.Dropdown(RECOGNIZER_LABELS, value=RECOGNIZER_LABELS[0], label='识别模型')
                    similarity_threshold = gr.Slider(0, 1, value=0.35, step=0.01, label='相似度阈值')
                compare_button = gr.Button('比较人脸', variant='primary')
                compare_summary = gr.Textbox(label='判断')
                compare_json = gr.Code(label='结构化结果', language='json')
                compare_button.click(
                    _compare,
                    [recognition_a, recognition_b, recognition_model, detector, provider, similarity_threshold],
                    [compare_summary, compare_json],
                )

            with gr.Tab('视频 / 跟踪'):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label='输入视频', sources=['upload', 'webcam'])
                        video_operation = gr.Dropdown(
                            ['人脸跟踪', '人脸检测', '匿名化'], value='人脸跟踪', label='视频任务'
                        )
                        video_blur = gr.Dropdown(
                            ['gaussian', 'pixelate', 'blackout', 'elliptical', 'median'],
                            value='pixelate',
                            label='匿名化方式',
                        )
                        video_run = gr.Button('处理视频', variant='primary')
                    with gr.Column():
                        video_output = gr.Video(label='输出视频')
                        video_json = gr.Code(label='处理统计', language='json')
                video_run.click(
                    _run_video,
                    [video_input, video_operation, detector, provider, detection_threshold, video_blur],
                    [video_output, video_json],
                )

            with gr.Tab('向量库 / 人脸检索'):
                gr.Markdown('先上传注册图片并按同一顺序逐行填写身份标签，再建立索引。索引路径位于运行机器本地。')
                with gr.Row():
                    with gr.Column():
                        enroll_files = gr.File(
                            label='注册图片（可多选）', file_count='multiple', file_types=['image'], type='filepath'
                        )
                        enroll_labels = gr.Textbox(label='身份标签（每行一个）', lines=6, placeholder='Alice\nBob')
                        db_path = gr.Textbox(
                            label='FAISS 索引目录', value=str(Path.cwd() / 'outputs' / 'uniface_vector_index')
                        )
                        vector_model = gr.Dropdown(RECOGNIZER_LABELS, value=RECOGNIZER_LABELS[0], label='识别模型')
                        build_button = gr.Button('建立 / 覆盖索引', variant='primary')
                        build_status = gr.Textbox(label='索引状态')
                    with gr.Column():
                        query_image = gr.Image(label='检索图片', type='numpy', sources=['upload', 'webcam'])
                        vector_threshold = gr.Slider(0, 1, value=0.4, step=0.01, label='匹配阈值')
                        search_button = gr.Button('检索图片中的人脸', variant='primary')
                        search_output = gr.Image(label='检索结果', type='numpy')
                        search_json = gr.Code(label='匹配明细', language='json')
                build_button.click(
                    _build_index,
                    [enroll_files, enroll_labels, db_path, vector_model, detector, provider],
                    build_status,
                )
                search_button.click(
                    _search_index,
                    [query_image, db_path, vector_model, detector, provider, vector_threshold],
                    [search_output, search_json],
                )

            with gr.Tab('模型管理'):
                gr.Markdown('可提前下载模型，避免首次推理时等待。下载器会校验每个权重文件的 SHA-256。')
                model_selection = gr.Dropdown(
                    ALL_MODEL_LABELS,
                    multiselect=True,
                    label='选择要下载的模型',
                )
                with gr.Row():
                    select_all = gr.Button('全选')
                    clear_all = gr.Button('清空')
                    download_button = gr.Button('下载并校验', variant='primary')
                download_log = gr.Textbox(label='下载结果', lines=12)
                select_all.click(lambda: gr.Dropdown(value=ALL_MODEL_LABELS), outputs=model_selection)
                clear_all.click(lambda: gr.Dropdown(value=[]), outputs=model_selection)
                download_button.click(_download, model_selection, download_log)

        gr.Markdown(
            '说明：本界面调用本地 UniFace 源码；检测器和模型实例按配置缓存。'
            '向量检索需要安装 `faiss-cpu`，部分表情模型需要 PyTorch。'
        )
    return demo


def main() -> None:
    """CLI entry point for launching the GUI."""
    parser = argparse.ArgumentParser(description='Launch the UniFace Gradio GUI')
    parser.add_argument('--host', default='127.0.0.1', help='Bind host')
    parser.add_argument('--port', type=int, default=7860, help='Bind port')
    parser.add_argument('--share', action='store_true', help='Create a temporary public Gradio link')
    args = parser.parse_args()
    build_app().queue().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == '__main__':
    main()
