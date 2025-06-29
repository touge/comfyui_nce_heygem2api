import os
import tempfile
import requests
import torchaudio
import soundfile as sf
from urllib.parse import urljoin

from .heygem_client import HeygemApiClient
from .utils import audio_to_tensor, video_to_tensor

import folder_paths
temp_dir = folder_paths.get_temp_directory()
TEMP_DIR = os.path.join(temp_dir, 'heygem')
os.makedirs(os.path.join(TEMP_DIR, 'temp'), exist_ok=True)



print(f"🐍 Heygem Generate Video Api Node folder_paths:{folder_paths.get_temp_directory()}")

TEMP_DIR = os.path.join(folder_paths.get_temp_directory(), 'heygem')

print(f"🐍 folder_paths.get_temp_directory: {folder_paths.get_temp_directory()}")

os.makedirs(TEMP_DIR, exist_ok=True)
print(f"🐍 Heygem Generate Video Api Node Initialized, using temp dir: {TEMP_DIR}")

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
    RETURN_NAME = ("VIDEO",)  # Changed to VIDEO to match the output type
    FUNCTION     = "process"
    CATEGORY     = CATEGORY

    def process(self, ApiConfigure, character_name, audio):
        client = HeygemApiClient(ApiConfigure)
        character_name = self._normalize_character_name(character_name)

        audio_path = self._cache_audio(audio)
        task_code  = self._submit_generation_task(client, character_name, audio_path)
        result_rel_path = self._wait_for_video(client, task_code)
        video_tensor = self._download_and_decode_video(client, result_rel_path)

        return (video_tensor,)

    def _normalize_character_name(self, name: str) -> str:
        name = name.strip()
        if not name:
            raise ValueError("请在 `character_name` 中输入要使用的角色名称")
        return name

    def _cache_audio(self, audio_data) -> str:
        return audio_to_tensor(
            cache_dir=os.path.join(os.path.dirname(__file__), "cache"),
            audio_tensor=audio_data["waveform"],
            sample_rate=int(audio_data["sample_rate"]),
            filename_prefix="heygem_audio_cache_",
            audio_format=".wav"
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

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            temp_path = tmp.name
            try:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
            except Exception as e:
                raise RuntimeError(f"视频下载失败") from e

        try:
            return video_to_tensor(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# class NCEHeygem2apiSpeakers:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "ApiConfigure": ("ApiConfigure", {"forceInput": True}),
#             }
#         }

#     # 改成 STRING，右侧会渲染成多行文本框
#     RETURN_TYPES = ("STRING",)
#     FUNCTION     = "process"
#     CATEGORY     = CATEGORY

#     def process(self, ApiConfigure):
#         url, headers = _prepare_request(ApiConfigure, "list_speakers")
#         try:
#             resp = requests.get(url, headers=headers, timeout=(3, 15))
#             resp.raise_for_status()
#             speakers = resp.json().get("speakers", [])
#             print(f"[Speakers] raw list: {repr(speakers)}")
#         except Exception as e:
#             print(f"[Speakers] 请求异常: {e}")
#             speakers = []

#         # 用换行拼接，每个说话人占一行
#         text = "\n".join(speakers) or "（没有发现任何说话人）"
#         return (text,)

# class NCEHeygem2apiGenSpeaker:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "ApiConfigure": ("ApiConfigure", {"forceInput": True}),
#                 "audio":       ("AUDIO",       {"forceInput": True}),
#                 "speaker":     ("STRING",      {"default": "nick"}),
#                 "overwrite":   ("BOOLEAN",     {"default": False}),
#                 "prompt":      ("STRING",      {"multiline": True}),
#             }
#         }

#     RETURN_TYPES = ("STRING",)
#     FUNCTION     = "process"
#     CATEGORY     = CATEGORY

#     def process(self, ApiConfigure, audio, speaker, overwrite, prompt):
#         url, headers = _prepare_request(ApiConfigure, "register_speaker_upload")

#         # dump AUDIO dict to a temp WAV file
#         waveform    = audio["waveform"]  # [1, C, T]
#         sample_rate = int(audio["sample_rate"])
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         tmp_path = tmp.name; tmp.close()
#         torchaudio.save(tmp_path, waveform.squeeze(0), sample_rate)

#         files = {
#             "wav_file": (
#                 os.path.basename(tmp_path),
#                 open(tmp_path, "rb"),
#                 "audio/wav"
#             )
#         }
#         data = {
#             "speaker":     speaker,
#             "prompt_text": prompt,
#             "overwrite":   str(overwrite).lower()
#         }

#         try:
#             resp = requests.post(url, headers=headers, data=data, files=files, timeout=(5, 300))
#             resp.raise_for_status()
#             return (resp.text,)
#         except Exception as e:
#             return (f"请求异常：{e}",)
#         finally:
#             try: os.unlink(tmp_path)
#             except: pass



