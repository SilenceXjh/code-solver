"""
LLM 客户端抽象基类

所有 Agent 只依赖这个接口，不直接调用任何 SDK。
好处：
  - 单元测试时注入 MockLLMClient，完全不需要真实 API
  - 将来切换模型（OpenAI → vLLM → Anthropic）只改 Client 实现
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    role: str    # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMClient(ABC):
    """所有 LLM 客户端必须实现的接口"""

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        发送对话请求，返回模型回复。

        Args:
            messages: 对话历史，包含 system/user/assistant 消息
            temperature: 采样温度，0.0 为确定性输出
            max_tokens: 最大生成 token 数
            json_mode: 是否强制输出合法 JSON（部分模型支持）
        """
        ...

    def chat_simple(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """便捷方法：system + user 两条消息，直接返回文本内容"""
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        return self.chat(messages, temperature, max_tokens, json_mode).content