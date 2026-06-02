"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import time
import urllib.error
import urllib.request
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI, APIConnectionError, APITimeoutError, BadRequestError, NotFoundError

from ..config import Config
from ..utils.logger import get_logger


logger = get_logger('mirofish.llm_client')


JSON_ONLY_INSTRUCTION = (
    "You must return only a valid JSON object. "
    "Do not include markdown fences, explanations, or any text outside the JSON object."
)


def clean_response_content(content: Optional[str]) -> str:
    """清理模型输出中的思考内容和markdown标记"""
    if not content:
        return ""

    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?```\s*$', '', cleaned)
    return cleaned.strip()


def _extract_first_json_object(text: str) -> str:
    """从文本中提取第一个完整的JSON对象"""
    start = text.find('{')
    if start == -1:
        return text.strip()

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return text[start:].strip()


def parse_json_content(content: Optional[str]) -> Dict[str, Any]:
    """解析模型返回的JSON文本，必要时尝试从包裹文本中提取JSON对象"""
    cleaned = clean_response_content(content)
    if not cleaned:
        raise ValueError("LLM返回的JSON格式无效: <empty>")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(cleaned)
        if extracted and extracted != cleaned:
            return json.loads(extracted)
        raise ValueError(f"LLM返回的JSON格式无效: {cleaned}")


def _augment_messages_for_json(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    augmented = [dict(message) for message in messages]
    if augmented and augmented[0].get("role") == "system":
        augmented[0]["content"] = f"{augmented[0]['content'].rstrip()}\n\n{JSON_ONLY_INSTRUCTION}"
        return augmented

    return [{"role": "system", "content": JSON_ONLY_INSTRUCTION}, *augmented]


def _resolve_base_url(client: OpenAI, base_url: Optional[str]) -> str:
    if base_url:
        return str(base_url)
    return str(getattr(client, "base_url", ""))


def _is_local_ollama_base_url(base_url: str) -> bool:
    return ":11434" in base_url and ("127.0.0.1" in base_url or "localhost" in base_url)


def _is_local_lm_studio_base_url(base_url: str) -> bool:
    lowered = base_url.lower()
    return ":1234" in lowered and ("127.0.0.1" in lowered or "localhost" in lowered)


def _normalize_ollama_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    normalized = re.sub(r"/v1$", "", normalized)
    return normalized


def _is_retryable_ollama_native_error(detail: str, status_code: Optional[int] = None) -> bool:
    lowered = detail.lower()
    if status_code is not None and status_code >= 500:
        return True

    retryable_markers = (
        "eof",
        "timeout",
        "timed out",
        "connection reset",
        "broken pipe",
        "unexpected eof",
        "temporarily unavailable",
    )
    return any(marker in lowered for marker in retryable_markers)


def _local_ollama_unavailable_error(base_url: str) -> RuntimeError:
    normalized = _normalize_ollama_base_url(base_url)
    return RuntimeError(
        f"无法连接到本地 Ollama 服务: {normalized}。"
        "请先启动 `ollama serve` 或打开 Ollama 应用后重试。"
    )


def _fetch_ollama_models(base_url: str) -> List[str]:
    request = urllib.request.Request(
        url=f"{_normalize_ollama_base_url(base_url)}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise _local_ollama_unavailable_error(base_url) from error

    models = body.get("models") or []
    return [item.get("name") for item in models if item.get("name")]


def _local_ollama_missing_model_error(requested_model: str, available_models: List[str]) -> RuntimeError:
    if not available_models:
        return RuntimeError(
            f"本地 Ollama 中还没有可用模型。当前请求模型: {requested_model}。"
            f"请先运行 `ollama pull {requested_model}`，等待下载完成后重试。"
        )

    available = ", ".join(available_models[:8])
    return RuntimeError(
        f"本地 Ollama 中找不到模型: {requested_model}。"
        f"当前可用模型: {available}。"
        f"如果你正在下载模型，请等待下载完成后重试；否则运行 `ollama pull {requested_model}`。"
    )


def _resolve_ollama_model_name(base_url: str, requested_model: str) -> str:
    available_models = _fetch_ollama_models(base_url)
    if requested_model in available_models:
        return requested_model

    family = requested_model.split(":", 1)[0]
    family_matches = [
        name for name in available_models
        if name == family or name.startswith(f"{family}:")
    ]

    latest_alias = f"{family}:latest"
    if latest_alias in family_matches:
        return latest_alias
    if len(family_matches) == 1:
        return family_matches[0]
    if family_matches:
        return family_matches[0]

    raise _local_ollama_missing_model_error(requested_model, available_models)


def _request_ollama_native_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int],
    max_attempts: int = 3,
) -> Tuple[str, Optional[str]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
    }
    options: Dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload["options"] = options

    for attempt in range(max_attempts):
        request = urllib.request.Request(
            url=f"{_normalize_ollama_base_url(base_url)}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            if attempt < max_attempts - 1 and _is_retryable_ollama_native_error(detail, getattr(error, "code", None)):
                logger.warning(
                    "Ollama native chat attempt %s/%s failed with retryable HTTP error: %s",
                    attempt + 1,
                    max_attempts,
                    detail[:200],
                )
                time.sleep(attempt + 1)
                continue
            raise ValueError(f"Ollama原生chat调用失败: {detail}") from error
        except urllib.error.URLError as error:
            reason = str(getattr(error, "reason", error))
            if attempt < max_attempts - 1 and _is_retryable_ollama_native_error(reason):
                logger.warning(
                    "Ollama native chat attempt %s/%s failed with retryable transport error: %s",
                    attempt + 1,
                    max_attempts,
                    reason[:200],
                )
                time.sleep(attempt + 1)
                continue
            raise _local_ollama_unavailable_error(base_url) from error

    content = clean_response_content(body.get("message", {}).get("content"))
    finish_reason = body.get("done_reason")
    if not content:
        raise ValueError("LLM返回空响应")
    return content, finish_reason


def request_text_completion(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    actual_base_url = _resolve_base_url(client, base_url)
    actual_model = model
    if _is_local_ollama_base_url(actual_base_url):
        actual_model = _resolve_ollama_model_name(actual_base_url, model)
    is_local_lm_studio = _is_local_lm_studio_base_url(actual_base_url)

    kwargs: Dict[str, Any] = {
        "model": actual_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    if timeout is not None and timeout > 0:
        kwargs["timeout"] = timeout

    max_attempts = 2 if is_local_lm_studio else 1
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(**kwargs)
            content = clean_response_content(response.choices[0].message.content)
            finish_reason = response.choices[0].finish_reason

            if not content:
                if _is_local_ollama_base_url(actual_base_url):
                    return _request_ollama_native_chat(
                        base_url=actual_base_url,
                        model=actual_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                if is_local_lm_studio and finish_reason == "length" and attempt < max_attempts - 1:
                    logger.warning(
                        "LM Studio returned empty content with finish_reason=length (attempt %s/%s), retrying once",
                        attempt + 1,
                        max_attempts,
                    )
                    time.sleep(1)
                    continue

                raise ValueError("LLM返回空响应")

            return content, finish_reason

        except BadRequestError as error:
            lowered = str(error).lower()
            if _is_local_ollama_base_url(actual_base_url) and "does not support chat" in lowered:
                return _request_ollama_native_chat(
                    base_url=actual_base_url,
                    model=actual_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            if is_local_lm_studio and "model reloaded" in lowered and attempt < max_attempts - 1:
                logger.warning(
                    "LM Studio reported model reload (attempt %s/%s), retrying once",
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(2)
                continue

            last_error = error
            break

        except NotFoundError as error:
            if _is_local_ollama_base_url(actual_base_url):
                available_models = _fetch_ollama_models(actual_base_url)
                raise _local_ollama_missing_model_error(model, available_models) from error
            last_error = error
            break

        except APIConnectionError as error:
            if _is_local_ollama_base_url(actual_base_url):
                raise _local_ollama_unavailable_error(actual_base_url) from error
            last_error = error
            break

        except APITimeoutError as error:
            if is_local_lm_studio and attempt < max_attempts - 1:
                logger.warning(
                    "LM Studio request timed out (attempt %s/%s), retrying once",
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(1)
                continue
            raise TimeoutError(f"LLM请求超时（{timeout}s）") from error

    if last_error:
        raise last_error
    raise ValueError("LLM返回空响应")


def request_json_completion(
    *,
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[str, Optional[str]]:
    """获取JSON响应，对不稳定的response_format实现执行一次兼容性回退"""
    actual_base_url = _resolve_base_url(client, base_url)
    strategies = []

    if not _is_local_lm_studio_base_url(actual_base_url):
        strategies.append(
            {
                "messages": messages,
                "response_format": {"type": "json_object"},
            }
        )

    strategies.append(
        {
            "messages": _augment_messages_for_json(messages),
            "response_format": None,
        }
    )
    last_error = None

    for strategy in strategies:
        try:
            content, finish_reason = request_text_completion(
                client=client,
                model=model,
                messages=strategy["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=strategy["response_format"],
                base_url=base_url,
                timeout=timeout,
            )
            if content:
                return content, finish_reason
            last_error = ValueError("LLM返回空JSON响应")
        except Exception as error:
            last_error = error

    if last_error:
        raise last_error
    raise ValueError("LLM返回的JSON格式无效: <empty>")


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        self.request_timeout = request_timeout if request_timeout is not None else Config.LLM_REQUEST_TIMEOUT
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        content, _finish_reason = request_text_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        content, _finish_reason = request_json_completion(
            client=self.client,
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=self.base_url,
            timeout=self.request_timeout,
        )
        return parse_json_content(content)

