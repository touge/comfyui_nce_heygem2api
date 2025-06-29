# cache_audio.py

import os
import tempfile
import torchaudio
import torch
import imageio
import numpy as np


def audio_to_tensor(
    cache_dir: str,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: str = ".wav"
) -> str:
    """
    将一个音频 Tensor 缓存到磁盘，并返回生成的文件路径。

    参数:
      cache_dir       – 临时文件存放目录，会自动创建。
      audio_tensor    – 应为 [channels, samples] 的 torch.Tensor。
      sample_rate     – 音频采样率 (Hz)。
      filename_prefix – 临时文件前缀 (默认 "cached_audio_")。
      audio_format    – 文件后缀/格式 (默认 ".wav")。

    返回:
      写入磁盘后的临时文件绝对路径。

    异常:
      在无法写入文件或保存音频时，会抛出 RuntimeError。
    """
    os.makedirs(cache_dir, exist_ok=True)

    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False
        ) as tmp_file:
            temp_filepath = tmp_file.name

        if audio_tensor.device.type != "cpu":
            audio_tensor = audio_tensor.cpu()

        # 🛠 修复点：降维到 [channels, samples]
        if audio_tensor.ndim == 3:
            # 假设为 [1, C, S] 形式（常见于 ComfyUI）
            audio_tensor = audio_tensor.squeeze(0)

        if audio_tensor.ndim != 2:
            raise ValueError(f"audio_tensor 维度应为 [channels, samples]，但当前为 {audio_tensor.shape}")

        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath

    except (OSError, RuntimeError, ValueError) as e:
        raise RuntimeError(f"Error caching audio tensor: {e}") from e

def video_to_tensor(video_path: str) -> torch.Tensor:
    """
    读取本地视频文件并转为 torch.Tensor 格式，RGB，float32，[0, 1]。

    参数:
      video_path – 本地 mp4 文件路径。

    返回:
      Tensor: [frames, height, width, channels]，dtype=torch.float32

    异常:
      - FileNotFoundError：文件不存在
      - RuntimeError：读取失败或格式异常
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[video_to_tensor] 找不到视频文件: {video_path}")

    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        height, width = meta["size"]
        expected_channels = 3

        expected_shape = (height, width, expected_channels)

        def frame_generator(reader, expected_shape):
            for i, frame in enumerate(reader):
                if frame.ndim == 2:
                    print(f"[video_to_tensor] 第 {i} 帧为灰度图像 shape={frame.shape}，跳过。")
                    continue
                if frame.shape[-1] == 1:
                    print(f"[video_to_tensor] 第 {i} 帧通道为 1，跳过: shape={frame.shape}")
                    continue
                if frame.shape[-1] == 4:
                    frame = frame[:, :, :3]  # 去除 alpha 通道
                if frame.shape != expected_shape:
                    print(f"[video_to_tensor] 第 {i} 帧尺寸异常，跳过: shape={frame.shape}, 预期: {expected_shape}")
                    continue
                yield (frame.astype(np.float32) / 255.0)

        frames = list(frame_generator(reader, expected_shape))
        if len(frames) == 0:
            raise RuntimeError(f"[video_to_tensor] 无法从 '{video_path}' 中读取任何有效帧")

        try:
            frames_np = np.stack(frames, axis=0).copy()
        except Exception as stack_err:
            raise RuntimeError(f"[video_to_tensor] np.stack 出错: {stack_err}") from stack_err

        try:
            tensor = torch.from_numpy(frames_np).to(torch.float32)
        except Exception as tensor_err:
            raise RuntimeError(f"[video_to_tensor] 转换为 tensor 失败: {tensor_err}") from tensor_err

        return tensor

    finally:
        reader.close()
