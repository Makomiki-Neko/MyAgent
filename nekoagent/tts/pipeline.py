"""TTS 后处理管道：消息文本清洗 + 语音生成。"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from nekoagent.config.loader import get_config
from nekoagent.llm.factory import get_summary_llm
from nekoagent.observability.logging import get_logger
from nekoagent.tts.engine import get_engine

_log = get_logger("tts.pipeline")


def extract_speech_text(text: str) -> str:
    """用总结 LLM 把含 markdown/表情/动作描述的回复清洗为纯 TTS 文本。"""
    try:
        cfg = get_config()
        llm = get_summary_llm(cfg)
        prompt = (
            "你是一个语音合成文本提取器。请从以下内容中提取适合朗读的纯净口头文本：\n"
            "1. 去掉 markdown 标记、代码块、列表符号\n"
            "2. 去掉动作描述（如「摸摸头」「脸红」「微笑」）和表情符号\n"
            "3. 保留口语化的表达、语气词\n"
            "4. 不许重述、改动原来的语言文本，必须保持原有的表述，仅去除不相关、不适合语音表述的部分\n"
            "5. 仅输出文本本身，不要任何解释或额外内容\n"
            "\n示例：\n"
            "输入：（猫尾轻轻甩动，耳朵微微向后抿）呜…Miya这不是在等主人嘛！｀へ´ノ 刚刚处理完今天的算法优化笔记，本来想早点休息的～\n"
            "输出：呜…Miya这不是在等主人嘛！刚刚处理完今天的算法优化笔记，本来想早点休息的\n"
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=text),
        ]
        resp = llm.invoke(messages)
        result = resp.content if hasattr(resp, "content") else str(resp)
        result = result.strip()
        if not result:
            return text[:200]
        return result[:500]
    except Exception as exc:
        _log.warning("TTS 文本提取 LLM 失败，回退简单处理: %s", exc)
        cleaned = text.strip()
        for ch in "*#`~>|-[]()!@#$%^&":
            cleaned = cleaned.replace(ch, "")
        return cleaned[:200] if cleaned.strip() else "嗯嗯"


def generate_tts_for_message(msg_id: int, text: str) -> None:
    """在后台线程生成 TTS 音频，存为 voice/sound/{msg_id}.wav。"""
    if not text or not text.strip():
        return
    try:
        speech_text = extract_speech_text(text)
        _log.info("TTS 提取结果 (%d): %r", msg_id, speech_text[:60])
        engine = get_engine()
        engine.generate_speech(speech_text, msg_id=msg_id)
    except Exception as exc:
        _log.warning("TTS 后处理失败 (msg_id=%d): %s", msg_id, exc)


def trigger_tts_async(msg_id: int, text: str) -> None:
    """异步触发 TTS，不阻塞调用方。"""
    threading.Thread(target=generate_tts_for_message, args=(msg_id, text), daemon=True).start()
