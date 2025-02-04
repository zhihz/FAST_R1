from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
import httpx
import json
import os
from pathlib import Path
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import asyncio

app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

# 设置模板
templates = Jinja2Templates(directory="templates")

# 配置文件路径
config_dir = Path("config")
config_file = config_dir / 'config.json'
key_file = config_dir / 'key'

# 确保配置目录存在
config_dir.mkdir(parents=True, exist_ok=True)

# 创建异步 HTTP 客户端
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(60.0, connect=30.0, read=None),  # 设置超时时间：总超时60秒，连接超时30秒，读取不设超时
    transport=httpx.AsyncHTTPTransport(retries=3)  # 设置重试次数
)

# 初始化加密密钥
def get_encryption_key():
    if not key_file.exists():
        # 生成一个随机的盐值
        salt = os.urandom(16)
        # 使用 PBKDF2 生成密钥
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
        # 保存密钥和盐值
        with open(key_file, 'wb') as f:
            f.write(salt + key)
    else:
        # 读取已存在的密钥
        with open(key_file, 'rb') as f:
            data = f.read()
            key = data[16:]  # 前16字节是盐值
    return Fernet(key)

# 获取加密器
fernet = get_encryption_key()

# 初始化配置
config = {
    'api_key': None,
    'api_key_set': False,
    'api_key_verified': False,
    'inference_model': None,
    'execution_model': None,
    'available_models': [],
    'inference_prompt': None,
    'execution_prompt': None
}

# 加载配置
def load_config():
    global config
    if config_file.exists():
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            if loaded_config.get('api_key'):
                # 解密 API 密钥
                try:
                    decrypted_key = fernet.decrypt(loaded_config['api_key'].encode()).decode()
                    config['api_key'] = decrypted_key
                    config['api_key_set'] = True
                    config['api_key_verified'] = loaded_config.get('api_key_verified', False)
                except Exception:
                    config['api_key'] = None
                    config['api_key_set'] = False
                    config['api_key_verified'] = False
            config['inference_model'] = loaded_config.get('inference_model')
            config['execution_model'] = loaded_config.get('execution_model')
            config['available_models'] = loaded_config.get('available_models', [])
            config['inference_prompt'] = loaded_config.get('inference_prompt')
            config['execution_prompt'] = loaded_config.get('execution_prompt')

# 保存配置
def save_config():
    save_data = config.copy()
    if save_data['api_key']:
        # 加密 API 密钥
        encrypted_key = fernet.encrypt(save_data['api_key'].encode()).decode()
        save_data['api_key'] = encrypted_key
    with open(config_file, 'w') as f:
        json.dump(save_data, f)

# 验证 API 密钥
async def verify_api_key(api_key: str) -> bool:
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = await http_client.get(
            "https://api.siliconflow.cn/v1/models",
            headers=headers,
            params={"type": "text", "sub_type": "chat"}
        )
        response.raise_for_status()
        return True
    except Exception:
        return False

# 加载已存在的配置
load_config()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": [],
            "api_key_set": config['api_key_set'],
            "api_key_verified": config['api_key_verified'],
            "inference_model": config['inference_model'],
            "execution_model": config['execution_model'],
            "available_models": config['available_models']
        }
    )

@app.post("/set-api-key")
async def set_api_key(request: Request, api_key_input: str = Form(...)):
    # 验证 API 密钥
    is_valid = await verify_api_key(api_key_input)
    
    if not is_valid:
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": [{"role": "assistant", "content": "API 密钥验证失败，请检查密钥是否正确"}],
                "api_key_set": False,
                "api_key_verified": False,
                "inference_model": config['inference_model'],
                "execution_model": config['execution_model'],
                "available_models": config['available_models']
            }
        )

    # 保存并验证 API 密钥
    config['api_key'] = api_key_input
    config['api_key_set'] = True
    config['api_key_verified'] = True
    
    # 获取可用模型列表
    try:
        headers = {"Authorization": f"Bearer {api_key_input}"}
        response = await http_client.get(
            "https://api.siliconflow.cn/v1/models",
            headers=headers,
            params={"type": "text", "sub_type": "chat"}
        )
        response.raise_for_status()
        models_data = response.json()
        config['available_models'] = models_data.get('data', [])
        save_config()
    except Exception as e:
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": [{"role": "assistant", "content": f"获取模型列表失败：{str(e)}"}],
                "api_key_set": config['api_key_set'],
                "api_key_verified": config['api_key_verified'],
                "inference_model": config['inference_model'],
                "execution_model": config['execution_model'],
                "available_models": config['available_models']
            }
        )

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": [],
            "api_key_set": config['api_key_set'],
            "api_key_verified": config['api_key_verified'],
            "inference_model": config['inference_model'],
            "execution_model": config['execution_model'],
            "available_models": config['available_models']
        }
    )

@app.post("/select-model")
async def select_model(
    request: Request, 
    inference_model: str = Form(...),
    execution_model: str = Form(...)
):
    if not config['api_key_verified']:
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": [{"role": "assistant", "content": "请先设置并验证 API 密钥"}],
                "api_key_set": config['api_key_set'],
                "api_key_verified": config['api_key_verified'],
                "inference_model": config['inference_model'],
                "execution_model": config['execution_model'],
                "available_models": config['available_models']
            }
        )

    config['inference_model'] = inference_model
    config['execution_model'] = execution_model
    save_config()
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "messages": [],
            "api_key_set": config['api_key_set'],
            "api_key_verified": config['api_key_verified'],
            "inference_model": config['inference_model'],
            "execution_model": config['execution_model'],
            "available_models": config['available_models']
        }
    )

async def chat_with_model(api_key: str, model: str, messages: list) -> dict:
    """调用 SiliconFlow API 进行对话"""
    # 处理消息中的系统提示语格式
    for msg in messages:
        if msg['role'] == 'system' and isinstance(msg['content'], str):
            content = msg['content'].strip()
            # 如果提示语以三引号开始和结束，去除它们
            if content.startswith('"""') and content.endswith('"""'):
                msg['content'] = content[3:-3].strip()
            elif content.startswith("'''") and content.endswith("'''"):
                msg['content'] = content[3:-3].strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,  # 启用流式传输
        "max_tokens": 2048,  # 增加 max_tokens
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1
    }
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            async with http_client.stream(
                "POST",
                "https://api.siliconflow.cn/v1/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    if not line.startswith("data: "):  # 跳过非data行
                        continue
                    
                    data = line[6:]  # 去掉 "data: " 前缀
                    if data == "[DONE]":
                        break
                        
                    try:
                        chunk = json.loads(data)
                        if chunk and 'choices' in chunk and chunk['choices']:
                            # 确保delta中的内容不为None
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content'] is None:
                                delta['content'] = ''
                            if 'reasoning_content' in delta and delta['reasoning_content'] is None:
                                delta['reasoning_content'] = ''
                            chunk['choices'][0]['delta'] = delta
                            yield chunk
                    except json.JSONDecodeError as e:
                        print(f"无法解析 JSON: {data}")
                        print(f"解析错误: {str(e)}")
                        continue
                return
                    
        except httpx.TimeoutException as e:
            retry_count += 1
            if retry_count == max_retries:
                error_msg = f"请求超时，已重试 {retry_count} 次，放弃重试。详细信息：{str(e)}"
                print(f"\n!!! 超时错误 !!!\n{error_msg}")
                raise Exception(error_msg)
            print(f"请求超时，{max_retries - retry_count} 次重试机会剩余")
            await asyncio.sleep(1)
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP 错误 {e.response.status_code}: {e.response.text}"
            print(f"\n!!! HTTP 错误 !!!\n{error_msg}")
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"其他错误: {str(e)}\n错误类型: {type(e).__name__}"
            print(f"\n!!! 未预期的错误 !!!\n{error_msg}")
            raise Exception(error_msg)

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    if not config['api_key_verified']:
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "请先设置并验证 API 密钥"}
                ],
                "api_key_set": config['api_key_set'],
                "api_key_verified": config['api_key_verified'],
                "inference_model": config['inference_model'],
                "execution_model": config['execution_model'],
                "available_models": config['available_models']
            }
        )

    if not (config['inference_model'] and config['execution_model']):
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "请先选择推理模型和执行模型"}
                ],
                "api_key_set": config['api_key_set'],
                "api_key_verified": config['api_key_verified'],
                "inference_model": config['inference_model'],
                "execution_model": config['execution_model'],
                "available_models": config['available_models']
            }
        )

    messages = []
    try:
        messages.append({"role": "user", "content": message})
        
        # 首先使用推理模型
        inference_response = await chat_with_model(
            config['api_key'],
            config['inference_model'],
            [{"role": "user", "content": message}]
        )
        
        inference_message = inference_response['choices'][0]['message']
        inference_content = inference_message['content']
        reasoning_content = inference_message.get('reasoning_content', '')
        
        # 显示推理模型的结果
        messages.append({
            "role": "assistant",
            "content": f"[推理模型] 回复：\n{inference_content}\n\n推理过程：\n{reasoning_content}"
        })
        
        # 然后使用执行模型处理推理结果，将完整的推理内容传递给执行模型
        execution_response = await chat_with_model(
            config['api_key'],
            config['execution_model'],
            [{
                "role": "user", 
                "content": f"基于以下推理过程生成回复：\n\n推理过程：{reasoning_content}\n\n推理结果：{inference_content}"
            }]
        )
        
        execution_content = execution_response['choices'][0]['message']['content']
        
        # 显示执行模型的结果
        messages.append({
            "role": "assistant",
            "content": f"[执行模型] 最终回复：\n{execution_content}"
        })
        
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": messages,
                "api_key_set": config['api_key_set'],
                "api_key_verified": config['api_key_verified'],
                "inference_model": config['inference_model'],
                "execution_model": config['execution_model'],
                "available_models": config['available_models']
            }
        )
    except Exception as e:
        error_msg = f"错误类型: {type(e).__name__}\n错误信息: {str(e)}"
        print(f"\n!!! 发生错误 !!!\n{error_msg}")
        messages.append({"role": "assistant", "content": f"错误：{error_msg}"})
        return templates.TemplateResponse(
            "chat.html",
            {
                "request": request,
                "messages": messages,
                "api_key_set": config['api_key_set'],
                "api_key_verified": config['api_key_verified'],
                "inference_model": config['inference_model'],
                "execution_model": config['execution_model'],
                "available_models": config['available_models']
            }
        )

@app.post("/save-prompt")
async def save_prompt(request: Request, type: str = Form(...), prompt: str = Form(...)):
    """保存系统提示语"""
    try:
        if type not in ['inference', 'execution']:
            return {"error": "无效的提示语类型"}
        
        # 处理提示语格式
        prompt = prompt.strip()
        
        # 如果提示语以三引号开始和结束，去除它们
        if prompt.startswith('"""') and prompt.endswith('"""'):
            prompt = prompt[3:-3].strip()
        elif prompt.startswith("'''") and prompt.endswith("'''"):
            prompt = prompt[3:-3].strip()
            
        # 更新配置
        config[f'{type}_prompt'] = prompt
        save_config()
        
        return {"success": True}
    except Exception as e:
        print(f"\n保存提示语出错: {str(e)}")
        return {"error": str(e)}

@app.post("/chat/stream")
async def chat_stream(request: Request, message: str = Form(...)):
    if not config['api_key_verified']:
        return {"error": "请先设置并验证 API 密钥"}

    if not (config['inference_model'] and config['execution_model']):
        return {"error": "请先选择推理模型和执行模型"}

    async def generate():
        try:
            # 首先使用推理模型
            inference_content = ""
            reasoning_content = ""
            
            # 发送阶段标记和模型信息
            yield f"data: {json.dumps({'phase': 'inference', 'model': config['inference_model']})}\n\n"
            
            # 使用保存的推理模型提示语
            inference_prompt = config.get('inference_prompt', "你需要仔细分析用户的问题和需求，运用逻辑推理能力，先进行假设和推演，然后再给出详细的解决方案和逻辑思路，但不用提供具体执行的代码。")
            
            async for chunk in chat_with_model(
                config['api_key'],
                config['inference_model'],
                [
                    {
                        "role": "system",
                        "content": inference_prompt
                    },
                    {
                        "role": "user",
                        "content": message
                    }
                ]
            ):
                if chunk and 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        inference_content += delta['content']
                    if 'reasoning_content' in delta:
                        reasoning_content += delta['reasoning_content']
                    yield f"data: {json.dumps(chunk)}\n\n"

            # 推理模型完成，发送特殊标记
            yield f"data: {json.dumps({'phase_complete': 'inference'})}\n\n"

            # 尝试解析推理结果为JSON
            try:
                # 提取JSON部分
                json_str = inference_content
                if '{' in inference_content:
                    json_str = inference_content[inference_content.find('{'):inference_content.rfind('}')+1]
                result = json.loads(json_str)
                
                if 'AIcount' in result:
                    ai_count = int(result['AIcount'])
                    execution_tasks = []
                    
                    # 创建所有执行AI的消息气泡
                    for i in range(1, ai_count + 1):
                        ai_key = f'AI{i}'
                        if ai_key in result:
                            yield f"data: {json.dumps({'phase': 'execution', 'model': config['execution_model'], 'ai_number': i})}\n\n"
                    
                    # 创建并发任务
                    async def execute_ai(ai_number, ai_task, response_queue):
                        try:
                            async for chunk in chat_with_model(
                                config['api_key'],
                                config['execution_model'],
                                [{
                                    "role": "system",
                                    "content": f"{config.get('execution_prompt', '你根据当前的回答和推理逻辑进行实际执行，比如生成代码')}\n\n你是 AI{ai_number}，你的具体任务是：{result[f'AI{ai_number}']}\n\n推理过程：{reasoning_content}"
                                },
                                {
                                    "role": "user",
                                    "content": message
                                }]
                            ):
                                if chunk and 'choices' in chunk and chunk['choices']:
                                    chunk['ai_number'] = ai_number
                                    await response_queue.put(f"data: {json.dumps(chunk)}\n\n")
                            
                            # 任务完成标记
                            await response_queue.put(f"data: {json.dumps({'phase_complete': 'execution', 'ai_number': ai_number})}\n\n")
                        except Exception as e:
                            print(f"AI{ai_number} 执行出错: {str(e)}")
                            await response_queue.put(f"data: {json.dumps({'error': f'AI{ai_number} 执行出错: {str(e)}'})}\n\n")
                    
                    # 创建响应队列和任务列表
                    response_queue = asyncio.Queue()
                    tasks = []
                    
                    # 创建所有AI的执行任务
                    for i in range(1, ai_count + 1):
                        if f'AI{i}' in result:
                            task = asyncio.create_task(execute_ai(i, result[f'AI{i}'], response_queue))
                            tasks.append(task)
                    
                    # 处理队列中的响应
                    active_tasks = len(tasks)
                    while active_tasks > 0:
                        try:
                            response = await response_queue.get()
                            yield response
                            if 'phase_complete' in response:
                                active_tasks -= 1
                        except Exception as e:
                            print(f"处理队列时出错: {str(e)}")
                            active_tasks -= 1
                    
                    # 等待所有执行任务完成
                    try:
                        await asyncio.gather(*tasks)
                    except Exception as e:
                        print(f"执行任务时出错: {str(e)}")
                        yield f"data: {json.dumps({'error': f'执行任务时出错: {str(e)}'})}\n\n"
                
                else:
                    # 如果不是JSON格式，使用普通的执行模式
                    yield f"data: {json.dumps({'phase': 'execution', 'model': config['execution_model']})}\n\n"
                    execution_prompt = config.get('execution_prompt', "你根据当前的回答和推理逻辑进行实际执行，比如生成代码")
                    async for chunk in chat_with_model(
                        config['api_key'],
                        config['execution_model'],
                        [{
                            "role": "system",
                            "content": execution_prompt
                        },
                        {
                            "role": "user", 
                            "content": f"基于以下推理过程生成回复：\n\n推理过程：{reasoning_content}\n\n推理结果：{inference_content}"
                        }]
                    ):
                        if chunk and 'choices' in chunk and chunk['choices']:
                            yield f"data: {json.dumps(chunk)}\n\n"
                    
                    yield f"data: {json.dumps({'phase_complete': 'execution'})}\n\n"
            
            except json.JSONDecodeError:
                # 如果解析JSON失败，使用普通的执行模式
                yield f"data: {json.dumps({'phase': 'execution', 'model': config['execution_model']})}\n\n"
                execution_prompt = config.get('execution_prompt', "你根据当前的回答和推理逻辑进行实际执行，比如生成代码")
                async for chunk in chat_with_model(
                    config['api_key'],
                    config['execution_model'],
                    [{
                        "role": "system",
                        "content": execution_prompt
                    },
                    {
                        "role": "user", 
                        "content": f"基于以下推理过程生成回复：\n\n推理过程：{reasoning_content}\n\n推理结果：{inference_content}"
                    }]
                ):
                    if chunk and 'choices' in chunk and chunk['choices']:
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                yield f"data: {json.dumps({'phase_complete': 'execution'})}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_msg = f"错误类型: {type(e).__name__}\n错误信息: {str(e)}"
            print(f"\n!!! 发生错误 !!!\n{error_msg}")
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 