import os
import json
import ast
import asyncio
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from client.response import StreamEvent, StreamEventType, TokenUsage, ToolCall, TextDelta

class LLMClient:
    def __init__(self, config=None):
        self.config = config
        self.base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
        self.api_key = "ollama"
        self.client = None

    def get_client(self):
        if self.client is None:
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        return self.client

    async def chat_completion(self, messages, tools=None, stream=True):
        client = self.get_client()
        model_name = "llama3.1"

        # SANITIZATION: Fix arguments that use single quotes (Python style) instead of double quotes (JSON style)
        sanitized_messages = []
        for msg in messages:
            m = dict(msg)
            if "tool_calls" in m and m["tool_calls"]:
                new_tool_calls = []
                for tc in m["tool_calls"]:
                    new_tc = dict(tc)
                    func = dict(new_tc["function"])
                    args = func.get("arguments")
                    
                    # Case 1: It's already a Dict -> Dump to JSON string
                    if isinstance(args, dict):
                        func["arguments"] = json.dumps(args)
                    # Case 2: It's a String, but might use single quotes
                    elif isinstance(args, str):
                        try:
                            # Test if it is already valid JSON
                            json.loads(args)
                        except json.JSONDecodeError:
                            # If not, try to parse it as a Python dict (handling single quotes)
                            try:
                                valid_dict = ast.literal_eval(args)
                                func["arguments"] = json.dumps(valid_dict)
                            except:
                                # If both fail, leave it alone (it might be broken or just a string)
                                pass
                                
                    new_tc["function"] = func
                    new_tool_calls.append(new_tc)
                m["tool_calls"] = new_tool_calls
            sanitized_messages.append(m)

        kwargs = {
            "model": model_name,
            "messages": sanitized_messages,
            "stream": stream,
            "temperature": 0.1,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = await client.chat.completions.create(**kwargs)
            
            if stream:
                async for chunk in response:
                    # Handle Text Content
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield StreamEvent(
                            type=StreamEventType.TEXT_DELTA,
                            text_delta=TextDelta(content=chunk.choices[0].delta.content)
                        )

                    # Handle Tool Calls
                    if chunk.choices and chunk.choices[0].delta.tool_calls:
                        tc = chunk.choices[0].delta.tool_calls[0]
                        if tc.function.name:
                            try:
                                args = json.loads(tc.function.arguments or "{}")
                            except:
                                args = tc.function.arguments
                                
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_COMPLETE,
                                tool_call=ToolCall(
                                    call_id=tc.id or f"call_{id(tc)}",
                                    name=tc.function.name,
                                    arguments=args
                                )
                            )

                yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, usage=None)
            else:
                content = response.choices[0].message.content
                yield StreamEvent(type=StreamEventType.TEXT_DELTA, text_delta=TextDelta(content=content))
                yield StreamEvent(type=StreamEventType.MESSAGE_COMPLETE, usage=None)

        except Exception as e:
            print(f"DEBUG: Error in chat_completion: {e}")
            yield StreamEvent(type=StreamEventType.ERROR, error=str(e))

    async def close(self):
        if self.client:
            await self.client.close()
            self.client = None
