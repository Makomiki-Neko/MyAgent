# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:36:52 2026

@author: Makomiki
"""

import time
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple

AudioLike = Union[
    str,  # wav path, URL, base64
    np.ndarray,  # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

DEVICE = "cuda"

model = Qwen3TTSModel.from_pretrained(
    "./QwenQwen3-TTS-12Hz-1.7B-Base",
    device_map=DEVICE,
    dtype=torch.bfloat16,
)

ref_audio = r"./ref_voice.mp3"
ref_text = "你好！我叫莉奈娅，是冒险家协会的顾问，这个小家伙是露米，是我的伙伴。\
    关于野外冒险你有什么想知道的，尽管问我好啦！对了对了，如果你在冒险时遇见了什么不可思议的生物，可别忘了告诉我哦！"


@torch.no_grad()
def clone_voice(
    self,
    ref_audio: AudioLike,
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行一次声音克隆，生成可复用的克隆状态，可选持久化保存到本地
    对应原方法中「参考音频编码 + 克隆提示生成」的重复计算部分

    Args:
        ref_audio: 参考音频输入
        ref_text: 参考音频对应的文本内容，可选
        x_vector_only_mode: 是否仅使用 x-vector 音色模式
        save_path: 本地保存路径，建议后缀为 .pt，为空则不保存

    Returns:
        clone_state: 可直接复用的克隆状态字典
    """
    # 生成克隆提示项（核心克隆计算，原方法每次都重复执行）
    prompt_items = self.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )
    # 转换为模型推理所需的字典格式
    voice_clone_prompt_dict = self._prompt_items_to_voice_clone_prompt(
        prompt_items
    )

    # 预计算参考文本的 token id，避免推理时重复分词
    ref_text_item = prompt_items[0].ref_text
    if ref_text_item is None or ref_text_item.strip() == "":
        ref_ids = None
    else:
        ref_ids = self._tokenize_texts([self._build_ref_text(ref_text_item)])[
            0
        ]

    # 打包完整克隆状态
    clone_state = {
        "voice_clone_prompt": voice_clone_prompt_dict,
        "ref_ids": ref_ids,
        "x_vector_only_mode": x_vector_only_mode,
    }

    # 持久化保存（张量移至 CPU，兼容跨设备加载）
    if save_path is not None:

        def _to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, list):
                return [_to_cpu(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            return obj

        save_state = _to_cpu(clone_state)
        torch.save(save_state, save_path)

    return clone_state


def load_clone_state(
    self,
    load_path: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    从本地文件加载克隆状态，并自动迁移到模型所在设备

    Args:
        load_path: 克隆状态文件路径（.pt）
        device: 目标设备，默认自动匹配模型当前设备

    Returns:
        加载完成的克隆状态字典
    """
    if device is None:
        device = next(self.model.parameters()).device

    # 加载并映射到目标设备
    clone_state = torch.load(
        load_path, map_location=device, weights_only=False
    )

    # 递归确保所有张量都在目标设备上
    def _to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, list):
            return [_to_device(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        return obj

    return _to_device(clone_state)


@torch.no_grad()
def generate_with_clone(
    self,
    text: Union[str, List[str]],
    clone_state: Union[Dict[str, Any], str],
    language: Union[str, List[str]] = None,
    non_streaming_mode: bool = False,
    **kwargs,
) -> Tuple[List[np.ndarray], int]:
    """
    基于已有的克隆状态执行推理，无需重复进行声音克隆
    完全兼容原 generate_voice_clone 的输出格式与额外参数

    Args:
        text: 生成文本，支持单条或批量
        clone_state: 克隆状态字典，或本地克隆文件路径
        language: 语言，支持单条或批量
        non_streaming_mode: 是否使用非流式生成
        **kwargs: 其他生成超参数，与原方法一致

    Returns:
        音频数组列表 + 采样率，与原方法输出完全一致
    """
    # 自动加载本地文件
    if isinstance(clone_state, str):
        clone_state = self.load_clone_state(clone_state)

    voice_clone_prompt_dict = clone_state["voice_clone_prompt"]
    ref_ids_base = clone_state["ref_ids"]

    # ========== 以下逻辑与原方法完全对齐 ==========
    # 处理文本与语言批量
    texts = self._ensure_list(text)
    languages = (
        self._ensure_list(language)
        if isinstance(language, list)
        else (
            [language] * len(texts)
            if language is not None
            else ["Auto"] * len(texts)
        )
    )
    if len(languages) == 1 and len(texts) > 1:
        languages = languages * len(texts)
    if len(texts) != len(languages):
        raise ValueError(
            f"Batch size mismatch: text={len(texts)}, language={len(languages)}"
        )

    self._validate_languages(languages)

    # 批量扩展：单克隆状态匹配多条文本
    ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
    if (
        ref_code_list is not None
        and len(ref_code_list) == 1
        and len(texts) > 1
    ):
        expanded_prompt = {}
        for k, v in voice_clone_prompt_dict.items():
            if isinstance(v, list) and len(v) == 1:
                expanded_prompt[k] = v * len(texts)
            else:
                expanded_prompt[k] = v
        voice_clone_prompt_dict = expanded_prompt
        ref_ids = (
            [ref_ids_base] * len(texts) if ref_ids_base is not None else None
        )
    else:
        ref_ids = (
            [ref_ids_base] * len(texts) if ref_ids_base is not None else None
        )

    # 构建输入与分词
    input_texts = [self._build_assistant_text(t) for t in texts]
    input_ids = self._tokenize_texts(input_texts)

    # 合并生成参数
    gen_kwargs = self._merge_generate_kwargs(**kwargs)

    # 模型生成
    talker_codes_list, _ = self.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=languages,
        non_streaming_mode=non_streaming_mode,
        **gen_kwargs,
    )

    # 音频解码 + 裁剪参考音频部分
    codes_for_decode = []
    for i, codes in enumerate(talker_codes_list):
        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
        if ref_code_list is not None and ref_code_list[i] is not None:
            codes_for_decode.append(
                torch.cat([ref_code_list[i].to(codes.device), codes], dim=0)
            )
        else:
            codes_for_decode.append(codes)

    wavs_all, fs = self.model.speech_tokenizer.decode(
        [{"audio_codes": c} for c in codes_for_decode]
    )

    wavs_out: List[np.ndarray] = []
    for i, wav in enumerate(wavs_all):
        ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
        if ref_code_list is not None and ref_code_list[i] is not None:
            ref_len = int(ref_code_list[i].shape[0])
            total_len = int(codes_for_decode[i].shape[0])
            cut = int(ref_len / max(total_len, 1) * wav.shape[0])
            wavs_out.append(wav[cut:])
        else:
            wavs_out.append(wav)

    return wavs_out, fs


"""
clone_state = clone_voice(
    self=model,
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
    save_path="./Linnea_Voice.pth",
)
"""

clone_state = load_clone_state(
    self=model, load_path="./Linnea_Voice.pth", device=DEVICE
)

s = time.time()

# 后续任意次数推理，无需再传参考音频
wavs1, sr = generate_with_clone(
    self=model,
    text="基于已有的克隆状态执行推理，无需重复进行声音克隆，完全兼容原 generate_voice_clone 的输出格式与额外参数",
    clone_state=clone_state,
    language="Chinese",
)


print(f"User Time {time.time() - s} Second.")

sf.write("output_voice_clone1.wav", wavs1[0], sr)
