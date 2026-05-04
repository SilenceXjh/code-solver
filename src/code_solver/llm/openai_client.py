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
        pricing: Optional[dict] = None,
        max_retries: int = 1,
        retry_delay: float = 2.0,
    ):
        """
        Args:
            model: 模型名称
            api_key: API 密钥，默认读取 DEEPSEEK_API_KEY 环境变量
            api_base: API 基础 URL，vLLM 部署时设置为本地地址
            max_retries: 请求失败时最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "empty")
        self.api_base = api_base  # None 表示使用 OpenAI 默认地址
        self.pricing = pricing or {
            "deepseek-v4-flash": {
                "input_per_1m_cache_hit": 0.0028,
                "input_per_1m_cache_miss": 0.14,
                "output_per_1m": 0.28,
            },
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None  # 懒加载
        self.reset_usage()

    def reset_usage(self) -> None:
        self._usage = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_tokens_cache_hit": 0,
            "input_tokens_cache_miss": 0,
            "cost_usd": 0.0,
            "unpriced_calls": 0,
        }

    def get_usage(self) -> dict:
        return {
            "calls": int(self._usage.get("calls", 0)),
            "input_tokens": int(self._usage.get("input_tokens", 0)),
            "input_tokens_cache_hit": int(self._usage.get("input_tokens_cache_hit", 0)),
            "input_tokens_cache_miss": int(self._usage.get("input_tokens_cache_miss", 0)),
            "output_tokens": int(self._usage.get("output_tokens", 0)),
            "cost_usd": float(self._usage.get("cost_usd", 0.0)),
            "unpriced_calls": int(self._usage.get("unpriced_calls", 0)),
        }

    def _resolve_pricing(self, model: str) -> Optional[tuple[float, float, float]]:
        if not self.pricing:
            return None
        
        spec = self.pricing.get(model)
        if not isinstance(spec, dict):
            return None

        input_per_1m_cache_hit = spec.get("input_per_1m_cache_hit")
        input_per_1m_cache_miss = spec.get("input_per_1m_cache_miss")
        output_per_1m = spec.get("output_per_1m")

        if input_per_1m_cache_hit is None or input_per_1m_cache_miss is None or output_per_1m is None:
            return None
        return float(input_per_1m_cache_hit), float(input_per_1m_cache_miss), float(output_per_1m)


    def _estimate_cost_usd(self, model: str, input_tokens_cache_hit: int, input_tokens_cache_miss: int, output_tokens: int) -> Optional[float]:
        pricing = self._resolve_pricing(model)
        if pricing is None:
            return None
        input_per_1m_cache_hit, input_per_1m_cache_miss, output_per_1m = pricing
        return (input_tokens_cache_hit / 1_000_000.0) * input_per_1m_cache_hit + (input_tokens_cache_miss / 1_000_000.0) * input_per_1m_cache_miss + (output_tokens / 1_000_000.0) * output_per_1m

    def _record_usage(self, model: str, input_tokens_cache_hit: int, input_tokens_cache_miss: int, output_tokens: int, cost_usd: Optional[float]) -> None:
        self._usage["calls"] += 1
        self._usage["input_tokens"] += input_tokens_cache_hit + input_tokens_cache_miss
        self._usage["output_tokens"] += output_tokens
        self._usage["input_tokens_cache_hit"] += input_tokens_cache_hit
        self._usage["input_tokens_cache_miss"] += input_tokens_cache_miss
        if cost_usd is None:
            self._usage["unpriced_calls"] += 1
        else:
            self._usage["cost_usd"] += float(cost_usd)


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
                # print("raw response:", resp)
                model = getattr(resp, "model", "") or self.model
                usage = getattr(resp, "usage", None)
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                
                if self.model == "deepseek-v4-flash":
                    input_tokens_cache_hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
                    input_tokens_cache_miss = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
                    
                    assert input_tokens == input_tokens_cache_hit + input_tokens_cache_miss, f"input_tokens ({input_tokens}) != input_tokens_cache_hit ({input_tokens_cache_hit}) + input_tokens_cache_miss ({input_tokens_cache_miss})"
                    cost_usd = self._estimate_cost_usd(model, input_tokens_cache_hit, input_tokens_cache_miss, output_tokens)
                    self._record_usage(model, input_tokens_cache_hit, input_tokens_cache_miss, output_tokens, cost_usd)
                
                return LLMResponse(
                    content=resp.choices[0].message.content,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"LLM call failed after {self.max_retries} retries: {last_error}"
        )
