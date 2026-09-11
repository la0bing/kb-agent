import json
import re
from typing import AsyncGenerator

import torch
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from transformers import AutoModelForMultimodalLM, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-2B"

processor = None
tokenizer = None
model = None

CALL_RE = re.compile(r"<function=(.*?)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=(.*?)>\n?(.*?)\n?</parameter>", re.S)
THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def load():
    global processor, tokenizer, model
    if model is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        tokenizer = getattr(processor, "tokenizer", processor)
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
    return tokenizer, model


def generate(messages, tools=None, max_new_tokens=512):
    tok, mdl = load()
    enc = tok.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(mdl.device)
    with torch.inference_mode():
        out = mdl.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    used = enc["input_ids"].shape[-1]
    text = tok.decode(out[0][used:], skip_special_tokens=True)
    return THINK_RE.sub("", text).strip(), used, int(out.shape[-1]) - used


def ask(prompt, max_new_tokens=256):
    return generate([{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens)[0]


def tool_schemas(llm_request):
    out = []
    for tool in (llm_request.config.tools or []) if llm_request.config else []:
        for fn in getattr(tool, "function_declarations", None) or []:
            params = fn.parameters_json_schema
            if params is None and fn.parameters is not None:
                params = fn.parameters.model_dump(exclude_none=True, mode="json")
            out.append({
                "type": "function",
                "function": {
                    "name": fn.name,
                    "description": fn.description or "",
                    "parameters": params or {"type": "object", "properties": {}},
                },
            })
    return out


def to_messages(llm_request):
    messages = []
    system = llm_request.config.system_instruction if llm_request.config else None
    if system:
        messages.append({"role": "system", "content": system if isinstance(system, str) else str(system)})

    for content in llm_request.contents or []:
        role = "assistant" if content.role == "model" else content.role or "user"
        text = []
        calls = []
        for part in content.parts or []:
            if part.function_response is not None:
                messages.append({
                    "role": "tool",
                    "name": part.function_response.name,
                    "content": json.dumps(part.function_response.response, default=str),
                })
            elif part.function_call is not None:
                calls.append({
                    "type": "function",
                    "function": {
                        "name": part.function_call.name,
                        "arguments": dict(part.function_call.args or {}),
                    },
                })
            elif part.text:
                text.append(part.text)
        if text or calls:
            message = {"role": role, "content": "\n".join(text)}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
    return messages


def cast(value, schema):
    kind = (schema or {}).get("type")
    value = value.strip()
    try:
        if kind == "integer":
            return int(float(value))
        if kind == "number":
            return float(value)
        if kind == "boolean":
            return value.lower() in ("true", "1", "yes")
        if kind in ("object", "array"):
            return json.loads(value)
    except (ValueError, json.JSONDecodeError):
        pass
    return value


def parse_calls(text, schemas):
    by_name = {s["function"]["name"]: s["function"].get("parameters") or {} for s in schemas}
    calls = []
    for name, body in CALL_RE.findall(text):
        name = name.strip()
        props = by_name.get(name, {}).get("properties", {})
        args = {}
        for key, value in PARAM_RE.findall(body):
            key = key.strip()
            args[key] = cast(value, props.get(key))
        calls.append((name, args))
    return calls


class QwenLocal(BaseLlm):
    model: str = MODEL_ID

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        schemas = tool_schemas(llm_request)
        text, sent, made = generate(to_messages(llm_request), schemas)

        parts = []
        for name, args in parse_calls(text, schemas):
            parts.append(types.Part.from_function_call(name=name, args=args))

        if not parts:
            clean = CALL_RE.sub("", text).replace("<tool_call>", "").replace("</tool_call>", "").strip()
            parts.append(types.Part.from_text(text=clean or "I could not produce a response."))

        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=sent,
                candidates_token_count=made,
                total_token_count=sent + made,
            ),
        )
