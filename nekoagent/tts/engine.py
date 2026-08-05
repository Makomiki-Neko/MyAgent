"""TTS 引擎封装：模型加载、声音克隆、语音生成。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from nekoagent.observability.logging import get_logger

_log = get_logger("tts.engine")

_VOICE_CLONE_DIR = Path("voice/clone")
_VOICE_SOUND_DIR = Path("voice/sound")
_VOICE_CLONE_DIR.mkdir(parents=True, exist_ok=True)
_VOICE_SOUND_DIR.mkdir(parents=True, exist_ok=True)

_CLONE_INDEX_FILE = _VOICE_CLONE_DIR / "clones.json"

_engine_instance: "TTSEngine | None" = None


class TTSEngine:
    def __init__(self, model_path: str = "", device: str = "cuda", active_clone: str | None = None) -> None:
        self.model_path = model_path
        self.device = device
        self._model = None
        self._clones: dict[str, dict[str, Any]] = {}
        self._active_clone_name: str | None = None
        self._load_clone_index()
        if active_clone:
            self.set_active_clone(active_clone)

    # ── 模型管理 ──

    def ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_path or not Path(self.model_path).exists():
            _log.warning("TTS 模型路径未配置或不存在: %s", self.model_path)
            return False
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_path,
                device_map=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            _log.info("TTS 模型加载成功 device=%s path=%s", self.device, self.model_path)
            return True
        except Exception as exc:
            _log.warning("TTS 模型加载失败: %s", exc)
            return False

    # ── 克隆声音管理 ──

    def _load_clone_index(self) -> None:
        if _CLONE_INDEX_FILE.exists():
            try:
                self._clones = json.loads(_CLONE_INDEX_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._clones = {}
        else:
            self._clones = {}
        _log.info("已加载 %d 个克隆声音", len(self._clones))

    def _save_clone_index(self) -> None:
        _CLONE_INDEX_FILE.write_text(
            json.dumps(self._clones, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def list_clones(self) -> list[dict[str, Any]]:
        return [{"name": n, **v} for n, v in self._clones.items()]

    def get_active_clone(self) -> dict[str, Any] | None:
        if self._active_clone_name and self._active_clone_name in self._clones:
            return {"name": self._active_clone_name, **self._clones[self._active_clone_name]}
        return None

    def set_active_clone(self, name: str | None) -> None:
        if name is None or name in self._clones:
            self._active_clone_name = name
            _log.info("激活克隆声音: %s", name)

    # ── 声音克隆 ──

    def clone_voice(self, audio_path: str, text: str, name: str) -> bool:
        if not self.ensure_model():
            return False
        try:
            import torch
            ref_audio = audio_path
            ref_text = text
            prompt_items = self._model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text, x_vector_only_mode=False,
            )
            voice_clone_prompt_dict = self._model._prompt_items_to_voice_clone_prompt(prompt_items)

            ref_text_item = prompt_items[0].ref_text
            if ref_text_item is None or ref_text_item.strip() == "":
                ref_ids = None
            else:
                ref_ids = self._model._tokenize_texts(
                    [self._model._build_ref_text(ref_text_item)]
                )[0]

            clone_state = {
                "voice_clone_prompt": voice_clone_prompt_dict,
                "ref_ids": ref_ids,
                "x_vector_only_mode": False,
            }

            save_path = str(_VOICE_CLONE_DIR / f"{name}.pt")

            def _to_cpu(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu()
                elif isinstance(obj, list):
                    return [_to_cpu(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                return obj

            torch.save(_to_cpu(clone_state), save_path)
            self._clones[name] = {"path": save_path}
            self._save_clone_index()
            _log.info("克隆声音已保存: name=%s path=%s", name, save_path)
            return True
        except Exception as exc:
            _log.error("声音克隆失败: %s", exc)
            return False

    # ── 语音生成 ──

    def generate_speech(self, text: str, clone_name: str | None = None, msg_id: int | None = None) -> str | None:
        clone_name = clone_name or self._active_clone_name
        if not clone_name or clone_name not in self._clones:
            _log.warning("未选择克隆声音，无法生成语音")
            return None
        if not self.ensure_model():
            return None

        try:
            import torch
            clone_info = self._clones[clone_name]
            clone_state = torch.load(clone_info["path"], map_location=self.device, weights_only=False)

            def _to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(self.device)
                elif isinstance(obj, list):
                    return [_to_device(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: _to_device(v) for k, v in obj.items()}
                return obj

            clone_state = _to_device(clone_state)
            voice_clone_prompt_dict = clone_state["voice_clone_prompt"]
            ref_ids = clone_state["ref_ids"] if clone_state["ref_ids"] is not None else None

            texts = [text]
            languages = ["Chinese"]
            input_texts = [self._model._build_assistant_text(t) for t in texts]
            input_ids = self._model._tokenize_texts(input_texts)

            ref_code_list = voice_clone_prompt_dict.get("ref_code", None)
            if ref_code_list is not None and len(ref_code_list) == 1 and len(texts) > 1:
                expanded_prompt = {}
                for k, v in voice_clone_prompt_dict.items():
                    if isinstance(v, list) and len(v) == 1:
                        expanded_prompt[k] = v * len(texts)
                    else:
                        expanded_prompt[k] = v
                voice_clone_prompt_dict = expanded_prompt

            gen_kwargs = self._model._merge_generate_kwargs()
            talker_codes_list, _ = self._model.model.generate(
                input_ids=input_ids,
                ref_ids=[ref_ids] * len(texts) if ref_ids is not None else None,
                voice_clone_prompt=voice_clone_prompt_dict,
                languages=languages,
                non_streaming_mode=False,
                **gen_kwargs,
            )

            codes_for_decode = []
            for i, codes in enumerate(talker_codes_list):
                rc = voice_clone_prompt_dict.get("ref_code", None)
                if rc is not None and rc[i] is not None:
                    codes_for_decode.append(torch.cat([rc[i].to(codes.device), codes], dim=0))
                else:
                    codes_for_decode.append(codes)

            wavs_all, fs = self._model.model.speech_tokenizer.decode(
                [{"audio_codes": c} for c in codes_for_decode]
            )

            wavs_out = []
            for i, wav in enumerate(wavs_all):
                rc = voice_clone_prompt_dict.get("ref_code", None)
                if rc is not None and rc[i] is not None:
                    ref_len = int(rc[i].shape[0])
                    total_len = int(codes_for_decode[i].shape[0])
                    cut = int(ref_len / max(total_len, 1) * wav.shape[0])
                    wavs_out.append(wav[cut:])
                else:
                    wavs_out.append(wav)

            save_name = f"{msg_id}.wav" if msg_id else f"tts_{hash(text)}.wav"
            save_path = str(_VOICE_SOUND_DIR / save_name)
            import soundfile as sf
            sf.write(save_path, wavs_out[0], fs)
            _log.info("语音已生成: %s", save_path)
            return save_path
        except Exception as exc:
            _log.error("语音生成失败: %s", exc)
            return None

    def get_audio_path(self, msg_id: int) -> str | None:
        path = _VOICE_SOUND_DIR / f"{msg_id}.wav"
        return str(path.resolve()) if path.exists() else None


def get_engine() -> TTSEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TTSEngine()
    return _engine_instance


def init_engine(model_path: str = "", device: str = "cpu", active_clone: str | None = None) -> TTSEngine:
    global _engine_instance
    _engine_instance = TTSEngine(model_path=model_path, device=device, active_clone=active_clone)
    return _engine_instance
