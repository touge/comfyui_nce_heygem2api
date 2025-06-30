import os
import tempfile
import torch
import torchaudio
import imageio
import numpy as np
import uuid
from datetime import datetime
import shutil


def prepare_cache_dir(cache_dir: str):
    """
    确保 cache_dir 是一个全新的空目录：
      1. 如果目录已存在，则删除整个目录及其所有内容
      2. 然后重新创建该目录
    """
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print(f"[Heygem] 已删除旧缓存目录: {cache_dir}")
        except Exception as e:
            print(f"[Heygem] 删除缓存目录失败: {e}")
            raise RuntimeError(f"无法清空缓存目录: {e}")

    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Heygem] 成功创建缓存目录: {cache_dir}")
    except OSError as e:
        print(f"[Heygem] 创建缓存目录失败: {e}")
        raise RuntimeError(f"无法创建缓存目录: {e}")
    
def cache_audio(
    cache_dir: str,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: str = ".wav"
) -> str:
    # os.makedirs(cache_dir, exist_ok=True)

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

        if audio_tensor.ndim == 3:
            audio_tensor = audio_tensor.squeeze(0)
        if audio_tensor.ndim != 2:
            raise ValueError(f"audio_tensor 维度应为 [channels, samples]，但当前为 {audio_tensor.shape}")

        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath

    except (OSError, RuntimeError, ValueError) as e:
        raise RuntimeError(f"Error caching audio tensor: {e}") from e

def cache_video_bytes(
    video_bytes: bytes,
    cache_dir: str,
    filename_prefix: str = "cached_video_",
    video_format: str = ".mp4"
) -> str:
    # os.makedirs(cache_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = str(uuid.uuid4())[:8]
    filename = f"{filename_prefix}{timestamp}_{uid}{video_format}"
    file_path = os.path.join(cache_dir, filename)
    try:
        with open(file_path, "wb") as f:
            f.write(video_bytes)
        print(f"[cache_video_bytes] 视频已保存到: {file_path}")
    except Exception as e:
        raise RuntimeError(f"[cache_video_bytes] 视频保存失败: {e}") from e

    return file_path


def video_to_tensor(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    reader = None
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()

        width, height = meta['size']
        channels = 3

        def frame_processor_generator(imgio_reader, expected_shape):
            for i, frame in enumerate(imgio_reader):
                if frame.shape[-1] == 4 and expected_shape[-1] == 3:
                     frame = frame[:, :, :3]
                if frame.shape != expected_shape:
                    print(f"Warning: Frame {i} has unexpected shape {frame.shape}. Expected {expected_shape}. Skipping.")
                    continue

                frame_processed = (frame.astype(np.float32) / 255.0)

                yield frame_processed

        frame_shape = (height, width, channels)
        gen_instance = frame_processor_generator(reader, frame_shape)

        frames_np_flat = np.fromiter(
            gen_instance,
            np.dtype((np.float32, frame_shape))
        )

        num_loaded_frames = len(frames_np_flat)

        if num_loaded_frames == 0:
            reader.close()
            raise RuntimeError(f"Failed to load any frames from video '{video_path}'. Check video file.")

        frame_shape = (height, width, channels)
        gen_instance = frame_processor_generator(reader, frame_shape)

        frames_structured_np = np.fromiter(
            gen_instance,
            dtype=np.dtype((np.float32, frame_shape))
        )
        print(f"Finished reading frames. Structured numpy array shape: {frames_structured_np.shape}, dtype: {frames_structured_np.dtype}")

        num_loaded_frames = len(frames_structured_np)

        if num_loaded_frames == 0:
            reader.close()
            raise RuntimeError(f"Failed to load any frames from video '{video_path}'. Check video file or its content.")

        total_scalars = num_loaded_frames * height * width * channels
        try:
            if frames_structured_np.size * frames_structured_np.dtype.itemsize != total_scalars * np.dtype(np.float32).itemsize:
                 pass

            frames_np = frames_structured_np.view(np.float32).reshape(-1, height, width, channels)
            print(f"Reshaped numpy array shape: {frames_np.shape}, dtype: {frames_np.dtype}")

        except Exception as reshape_e:
            print(f"Error reshaping numpy array after fromiter: {reshape_e}")
            reader.close()
            raise RuntimeError(f"Failed to reshape frame data loaded from '{video_path}'. Likely mismatch in expected frame dimensions or data.") from reshape_e

        try:
            tensor = torch.from_numpy(frames_np)
            print("Tensor conversion successful.")
        except Exception as tensor_e:
            print(f"Error converting numpy array to torch tensor: {tensor_e}")
            reader.close()
            raise RuntimeError(f"Failed to convert numpy array to torch tensor for '{video_path}'. Likely out of memory.") from tensor_e

        reader.close()

        print(f"Successfully loaded video '{video_path}' into tensor with shape {tensor.shape}.")
        return tensor
    except Exception as e:
        if reader is not None:
            reader.close()
        raise RuntimeError(f"Failed to read video '{video_path}': {e}") from e
