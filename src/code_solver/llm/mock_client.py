"""
Mock LLM 客户端，专用于单元测试

支持两种模式：
  1. FixedMockClient：始终返回同一个预设响应
  2. ScriptedMockClient：按调用顺序返回不同响应（测试多轮对话场景）
  3. FnMockClient：用函数动态生成响应（最灵活）
"""

from code_solver.llm.base import LLMClient, LLMResponse, Message

class FixedMockClient(LLMClient):
    """始终返回同一个预设字符串"""

    def __init__(self, response: str, model: str = "mock"):
        self.response = response
        self.model = model
        self.call_count = 0
        self.call_history: list[list[Message]] = []

    def reset_usage(self) -> None:
        self.call_count = 0
        self.call_history = []

    def get_usage(self) -> dict:
        return {
            "calls": int(self.call_count),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "unpriced_calls": 0,
        }

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.call_count += 1
        self.call_history.append(messages)
        return LLMResponse(content=self.response, model=self.model)


class ScriptedMockClient(LLMClient):
    """
    按调用顺序依次返回预设响应列表。
    列表用完后重复最后一个（避免 IndexError）。
    """

    def __init__(self, responses: list[str], model: str = "mock"):
        assert responses, "responses 不能为空"
        self.responses = responses
        self.model = model
        self.call_count = 0
        self.call_history: list[list[Message]] = []

    def reset_usage(self) -> None:
        self.call_count = 0
        self.call_history = []

    def get_usage(self) -> dict:
        return {
            "calls": int(self.call_count),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "unpriced_calls": 0,
        }

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_history.append(messages)
        self.call_count += 1
        return LLMResponse(content=self.responses[idx], model=self.model)


class FnMockClient(LLMClient):
    """用函数动态生成响应，函数签名：(messages: list[Message]) -> str"""

    def __init__(self, fn, model: str = "mock"):
        self.fn = fn
        self.model = model
        self.call_count = 0
        self.call_history: list[list[Message]] = []

    def reset_usage(self) -> None:
        self.call_count = 0
        self.call_history = []

    def get_usage(self) -> dict:
        return {
            "calls": int(self.call_count),
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "unpriced_calls": 0,
        }

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.call_history.append(messages)
        self.call_count += 1
        content = self.fn(messages)
        return LLMResponse(content=content, model=self.model)
