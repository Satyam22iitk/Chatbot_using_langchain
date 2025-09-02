from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional, Dict
from deepseek import DeepSeek

class DeepSeekLLM(BaseLLM):
    model_name: str = "deepseek-chat"
    api_key: str
    temperature: float = 0
    streaming: bool = False
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        client = DeepSeek(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            stream=self.streaming
        )
        
        if self.streaming:
            # Handle streaming response
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    if run_manager:
                        run_manager.on_llm_new_token(token)
            return full_response
        else:
            return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "deepseek"
