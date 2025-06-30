import os
import tempfile
import requests
import torchaudio
import soundfile as sf
from urllib.parse import urljoin
import uuid
from datetime import datetime

from .heygem_client import HeygemApiClient
from .utils import cache_audio, video_to_tensor, cache_video_bytes, prepare_cache_dir

cache_dir = os.path.join(os.path.dirname(__file__), "cache")
prepare_cache_dir(cache_dir)

CATEGORY = "🐍 NCE/Heygem4API"

class NCEHeygemConfigure:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_host": ("STRING", {"default": "http://192.168.0.253"}),
                "api_port": ("INT", {"default": 8001, "min": 1, "max": 65535}),
                "api_key":  ("STRING", {"default": "your-secret-key"}),
            }
        }

    RETURN_TYPES = ("ApiConfigure",)
    FUNCTION     = "process"
    CATEGORY     = CATEGORY

    def process(self, api_host, api_port, api_key):
        host = api_host.rstrip("/")
        base = f"{host}:{api_port}/" if host.startswith("http") else f"http://{host}:{api_port}/"
        return ({"api_base": base, "api_key": api_key},)

class NCEHeygemGenerateVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ApiConfigure":   ("ApiConfigure", {"forceInput": True}),
                "character_name": ("STRING",),
                "audio":          ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAME = ("VIDEO",)
    FUNCTION     = "process"
    CATEGORY     = CATEGORY

    def process(self, ApiConfigure, character_name, audio):
        try:
            client = HeygemApiClient(ApiConfigure)
            character_name = self._normalize_character_name(character_name)

            audio_path = self._cache_audio(audio)
            task_code = self._submit_generation_task(client, character_name, audio_path)
            result_rel_path = self._wait_for_video(client, task_code)
            video_tensor = self._download_and_decode_video(client, result_rel_path)

            return (video_tensor,)
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def _normalize_character_name(self, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("请在 `character_name` 中输入要使用的角色名称")
        return name

    def _cache_audio(self, audio_data) -> str:
        # cache_dir = os.path.join(os.path.dirname(__file__), "cache", "audio")
        return cache_audio(
            cache_dir= cache_dir,
            audio_tensor= audio_data["waveform"],
            sample_rate= int(audio_data["sample_rate"]),
            filename_prefix= "heygem_audio_cache_",
            audio_format= ".wav"
        )

    def _submit_generation_task(self, client: HeygemApiClient, character_name, audio_path) -> str:
        with open(audio_path, "rb") as audio_file:
            files = {
                "audio_file": (os.path.basename(audio_path), audio_file, "audio/wav")
            }
            data = {"character_name": character_name}
            response = client.post("generate-video", data=data, files=files, timeout=(5, 300))

        response.raise_for_status()
        task_code = response.json().get("task_code")
        if not task_code:
            raise ValueError("生成接口未返回 task_code")
        return task_code

    def _wait_for_video(self, client: HeygemApiClient, task_code: str) -> str:
        response = client.get("generate-video-progress", params={"task_code": task_code}, timeout=None)
        response.raise_for_status()
        data = response.json()
        status = data.get("status", data.get("detail", {}).get("status"))
        if status != 2:
            raise RuntimeError(f"视频生成失败或未完成，status={status}")

        result_path = data.get("detail", {}).get("result")
        if not result_path:
            raise ValueError("进度接口未返回 result 字段")
        return result_path

    def _download_and_decode_video(self, client: HeygemApiClient, result_path: str):
        response = client.get(f"video?path={result_path}", stream=True)
        response.raise_for_status()

        # 统一缓存目录
        # cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        video_bytes = b''.join(response.iter_content(chunk_size=8192))
        print(f"[Heygem] 下载视频字节流成功，大小: {len(video_bytes)} bytes")
        video_path = cache_video_bytes(
            video_bytes= video_bytes,
            cache_dir= cache_dir,
        )
        try:
            return video_to_tensor(video_path)
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

class NCEHeygemCharacters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ApiConfigure": ("ApiConfigure", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, ApiConfigure):
        client = HeygemApiClient(ApiConfigure)
        try:
            resp = client.get("characters", timeout=(3, 15))
            resp.raise_for_status()
            characters = resp.json()
            print(f"[Characters] raw list: {repr(characters)}")
        except Exception as e:
            print(f"[Characters] 请求异常: {e}")
            characters = []

        # 拼接 character_name 字段
        names = [c.get("character_name", "") for c in characters if isinstance(c, dict)]
        result_text = "\n".join(names) if names else "（没有发现任何角色）"
        return (result_text,)


class NCEHeygemUploadCharacter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ApiConfigure":   ("ApiConfigure", {"forceInput": True}),
                "character_name": ("STRING", {"default": "老师"}),
                "video":          ("VIDEO", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    def process(self, ApiConfigure, character_name, video):
        # ⬇️ 内联的辅助函数
        def _extract_video_file_path(video_obj):
            path = getattr(video_obj, "_VideoFromFile__file", None)
            return path if isinstance(path, str) else None

        client = HeygemApiClient(ApiConfigure)
        video_path = _extract_video_file_path(video)

        if not video_path or not os.path.exists(video_path):
            return (f"[UploadCharacter] 无效的视频文件路径: {video_path}",)

        if not character_name.strip():
            return (f"[UploadCharacter] 角色名称不能为空",)

        data = {
            "name": character_name
        }

        try:
            with open(video_path, "rb") as f:
                files = {
                    "video_file": (os.path.basename(video_path), f, "video/mp4")
                }
                resp = client.post("characters/upload", data=data, files=files, timeout=(5, 60))
                resp.raise_for_status()
                result = resp.json()
                return (f"上传成功：{result.get('character_name', character_name)}",)

        except Exception as e:
            return (f"[UploadCharacter] 上传失败：{e}",)
