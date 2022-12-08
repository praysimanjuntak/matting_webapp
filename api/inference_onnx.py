"""
python3 ./inference_onnx.py \
    --checkpoint "rvm_mobilenetv3_fp32.onnx" \
    --device cpu \
    --dtype fp32 \
    --input-source "input.mp4" \
    --downsample-ratio 0.25 \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple, Union, Type
from tqdm.auto import tqdm
import onnxruntime as ort
import cv2
import numpy as np

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter


def convert_video(session: ort.InferenceSession,
                  input_source: str,
                  background_image: str = None,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = 'cpu',
                  dtype: Type[Union[np.single, np.half]] = np.float32):
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: CUDA or not, Only need to manually provide if model is a ONNX freezed model.
        dtype: fp16 or fp32, Only need to manually provide if model is a ONNX freezed model.
    """

    assert downsample_ratio is None or (
            downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    assert output_video_mbps == None or output_type == 'video', 'Mbps is not available for png_sequence output.'

    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
        # auto_downsample_ratio before CUDA io binding
        if downsample_ratio is None:
            downsample_ratio = auto_downsample_ratio(*source[0].shape[1:])
    else:
        source = ImageSequenceReader(input_source, transform)
        if downsample_ratio is None:
            downsample_ratio = auto_downsample_ratio(*source[0].shape[1:])
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)

    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = VideoWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = VideoWriter(output_foreground, 'png')

    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.from_numpy(np.array([120, 255, 155], dtype=dtype)).div(255).view(1, 1, 3, 1, 1).to(device)

    try:

        bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)

        if device.lower() == "cuda":
            io = session.io_binding()
            # Create tensors on CUDA.
            rec = [ort.OrtValue.ortvalue_from_numpy(np.zeros([1, 1, 1, 1], dtype=dtype), 'cuda')] * 4
            downsample_ratio = ort.OrtValue.ortvalue_from_numpy(np.asarray([downsample_ratio], dtype=np.float32),
                                                                'cuda')
            # Set output binding.
            for name in ['fgr', 'pha', 'r1o', 'r2o', 'r3o', 'r4o']:
                io.bind_output(name, 'cuda')
        else:
            io = None
            rec = [np.zeros([1, 1, 1, 1], dtype=dtype)] * 4  # Must match dtype of the model.
            downsample_ratio = np.array([downsample_ratio], dtype=np.float32)  # dtype always FP32

        for src in reader:
            src = src.cpu().numpy()  # torch.Tensor -> np.ndarray CPU [B,C,H,W] for ONNX file
            if device.lower() == "cuda" and io is not None:
                io.bind_cpu_input('src', src)
                io.bind_ortvalue_input('r1i', rec[0])
                io.bind_ortvalue_input('r2i', rec[1])
                io.bind_ortvalue_input('r3i', rec[2])
                io.bind_ortvalue_input('r4i', rec[3])
                io.bind_ortvalue_input('downsample_ratio', downsample_ratio)

                session.run_with_iobinding(io)

                fgr, pha, *rec = io.get_outputs()

                # Transfer `fgr`, `pha`, and 'bgr' to CPU.
                fgr = fgr.numpy()
                pha = pha.numpy()
                bgr = bgr.cpu()
            else:
                fgr, pha, *rec = session.run([], {
                    'src': src,
                    'r1i': rec[0],
                    'r2i': rec[1],
                    'r3i': rec[2],
                    'r4i': rec[3],
                    'downsample_ratio': downsample_ratio
                })

            fgr = torch.from_numpy(fgr).unsqueeze(1)  # [B,T=1,C=3,H,W]
            pha = torch.from_numpy(pha).unsqueeze(1)  # [B,T=1,C=1,H,W]

            if output_foreground is not None:
                writer_fgr.write(fgr[0])
            if output_alpha is not None:
                writer_pha.write(pha[0])
            if output_composition is not None:
                if output_type == 'video':
                    com = fgr * pha + bgr * (1 - pha)
                else:
                    fgr = fgr * pha.gt(0)
                    com = torch.cat([fgr, pha], dim=-3)
                writer_com.write(com[0])

            bar.update(1)  # T=1
            # percentage = bar.n / bar.total * 100
            # await ws.send_text(str(percentage))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()

        if background_image is not None:
            alpha = cv2.VideoCapture("alpha.mp4")
            composition = cv2.VideoCapture(input_source)
            length = int(composition.get(cv2.CAP_PROP_FRAME_COUNT))
            image = cv2.imread(background_image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            width = int(composition.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
            height = int(composition.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
            size = (width, height)
            out = cv2.VideoWriter("composed.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps=frame_rate, frameSize=size)
            print("Getting composed_video...")
            with tqdm(total=length) as pbar:
                while True:
                    ret, alpha_frame = alpha.read()
                    ret2, composition_frame = composition.read()
                    if ret:
                        mask = cv2.inRange(alpha_frame, np.array([0, 0, 0]), np.array([130, 130, 130]))
                        background_image = cv2.resize(image, (composition_frame.shape[1], composition_frame.shape[0]) )

                        composition_frame[mask != 0] = [0, 0, 0]
                        background_image[mask == 0 ] = [0, 0, 0]

                        complete_image = background_image + composition_frame
                        out.write(complete_image)
                        pbar.update(1)
                    else:
                        break

            out.release()
            alpha.release()

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, checkpoint: str, device: str, dtype: str):
        assert dtype in ("fp16", "fp32")
        self.session = ort.InferenceSession(checkpoint, providers=['CUDAExecutionProvider'])
        self.device = device
        self.dtype = np.float32 if dtype == "fp32" else np.float16

    def convert(self, *args, **kwargs):
        convert_video(self.session, device=self.device, dtype=self.dtype, *args, **kwargs)