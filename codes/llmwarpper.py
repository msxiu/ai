''' 使用OpenAI的 chat.completions.create 创建远程请求提交模型处理 '''
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json

# 1. 定义模型封装类
class WeatherLLMWarpper:
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "qwen3:14b", api_key: str = "not-reeded", prompt:str = "", tools = []):
        """ 
        初始化 DeepSeek 包装器

        Args:
            api_key: DeepSeek API密钥
            model: 模型名称
            base_url: API基础URL 
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.tools: List[Dict[str, Any]] = []
        self.tool_handlers: Dict[str, callable] = {}
        self.system_prompt: Optional[str] = None
        self.system_prompt = prompt
        for item in tools:
            desc = item["description"]
            self.tools.append(desc)
            keyname = desc["function"]["name"]
            self.tool_handlers[keyname] = item["executor"]

    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """ 
        执行工具调用

        Args:
            tool_call: 工具调用信息
        Returns:
            str: 工具执行结果   
        """
        tool_name = tool_call.get("function", {}).get("name")
        arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        if tool_name in self.tool_handlers:
            try:
                result = self.tool_handlers[tool_name](**arguments)
                return json.dumps({"result": result, "status": "success"})
            except Exception as e:
                return json.dumps({"error": str(e), "status": "failed"})
        else:
            return json.dumps({"error": f"Tool {tool_name} not found", "status": "failed"})
    
    def invoke(self, prompt: str, auto_execute_tools: bool = True, max_tool_iterations: int = 5) -> str:
        """ 
        调用模型

        Args:
            prompt: 用户提示词
            auto_execute_tools: 是否自动执行工具
            max_tool_iterations: 最大工具调用迭代次数
        Returns:
            str: 模型响应 
        """
        messages = []
        if self.system_prompt:  # 添加系统提示词
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        iteration = 0
        while iteration < max_tool_iterations: # 调用API
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=messages, 
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None
            )
            message = response.choices[0].message
            messages.append(message.model_dump())
            if not auto_execute_tools or not message.tool_calls:  # 检查是否需要调用工具
                return message.content or ""
            for tool_call in message.tool_calls:  # 执行工具调用
                tool_result = self._execute_tool(tool_call.model_dump())
                messages.append({ "role": "tool", "tool_call_id": tool_call.id, "content": tool_result })
            iteration += 1
        # 如果达到最大迭代次数，获取最终响应
        final_response = self.client.chat.completions.create(model=self.model, messages=messages)
        return final_response.choices[0].message.content or ""
    
    def stream_invoke(self, prompt: str):
        """ 
        流式调用模型
        Args:
            prompt: 用户提示词
        Yields:
            str: 流式响应片段 
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        stream = self.client.chat.completions.create(
            model=self.model, 
            messages=messages, 
            tools=self.tools if self.tools else None,
            tool_choice="auto" if self.tools else None, stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# 2. 定义工具
def get_weather(city: str) -> str:
    """ 获取天气信息 """
    weather_data = { "北京": "晴天，25°C", "上海": "多云，28°C", "广州": "雷阵雨，30°C" }
    return weather_data.get(city, f"未知城市: {city}")
def calculate(expression: str) -> float:
    """计算数学表达式"""
    try: # 注意：实际使用时请使用安全的计算方式
        return eval(expression)
    except Exception as e:
        return f"计算错误: {str(e)}"

# 3. 创建包装器实例
wrapper = WeatherLLMWarpper(base_url="http://localhost:11434/v1", model="qwen3:14b", api_key="your-deepseek-api-key",
    prompt = "你是一个智能助手，可以使用工具来帮助用户解决问题。",
    tools = [{
        "executor":get_weather,
        "description":{  
            "type": "function",
            "function": { 
                "name": "get_weather", 
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称，如：北京、上海" }
                    },
                    "required": ["city"]
                }
            }
        }
    }, { 
        "executor":calculate,
        "description":{ 
            "type": "function",
            "function": { 
                "name": "calculate", 
                "description": "计算数学表达式",
                "parameters": { 
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式，如：'2 + 3 * 4'" }
                    },
                    "required": ["expression"]
                }
            }
        }
    }])

# 4. 使用示例， 自动执行工具
response = wrapper.invoke("北京今天天气怎么样？")
print(response)
# 多工具协同
response = wrapper.invoke("帮我计算 (25 + 35) * 2，然后告诉我这个结果是否适合今天广州的天气？")
print(response)

# 5. 流式输出
for chunk in wrapper.stream_invoke("介绍一下DeepSeek R1模型"):
    print(chunk, end="")