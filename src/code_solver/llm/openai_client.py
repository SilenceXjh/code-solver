"""
OpenAI 兼容的 LLM 客户端

支持：
  - OpenAI GPT 系列（gpt-4o, gpt-4o-mini 等）
  - vLLM 部署的开源模型（兼容 OpenAI API 格式）
  - 任何支持 OpenAI Chat Completions 协议的服务
"""

import os
import time
from typing import Optional

from code_solver.llm.base import LLMClient, LLMResponse, Message


class OpenAIClient(LLMClient):
    """
    OpenAI Chat Completions 客户端

    用法：
        client = OpenAIClient(model="gpt-4o-mini")
        resp = client.chat_simple("You are helpful.", "What is 2+2?")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Args:
            model: 模型名称
            api_key: API 密钥，默认读取 OPENAI_API_KEY 环境变量
            api_base: API 基础 URL，vLLM 部署时设置为本地地址
            max_retries: 请求失败时最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "empty")
        self.api_base = api_base  # None 表示使用 OpenAI 默认地址
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None  # 懒加载

    def _get_client(self):
        """懒加载 OpenAI 客户端，避免在 import 时就要求安装 openai"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "请安装 openai 库: pip install openai"
                )
            kwargs = {"api_key": self.api_key}
            if self.api_base:
                kwargs["base_url"] = self.api_base
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        openai_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
        kwargs = dict(
            model=self.model,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # print("llm input messages:", openai_messages)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(**kwargs)
                # print("llm response:", resp.choices[0].message.content)
                return LLMResponse(
                    content=resp.choices[0].message.content,
                    model=resp.model,
                    input_tokens=resp.usage.prompt_tokens,
                    output_tokens=resp.usage.completion_tokens,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_error}"
        )