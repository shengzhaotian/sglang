# 服务工具函数

本文档提供服务相关的工具函数，供主Skill使用。

## 服务就绪判断

```python
import requests
import time

def wait_for_service_ready(port, timeout=300):
    """等待服务就绪
    
    Args:
        port: 服务端口
        timeout: 超时时间（秒）
    
    Returns:
        bool: 服务是否就绪
    """
    url = f"http://localhost:{port}/v1/models"
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    
    return False
```

## 端口可用性检查

```python
import socket

def check_port_available(port):
    """检查端口是否可用
    
    Args:
        port: 端口号
    
    Returns:
        bool: 端口是否可用
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0
```

## 发送测试请求

```python
import requests

def send_test_request(port, messages, max_tokens=50):
    """发送测试请求
    
    Args:
        port: 服务端口
        messages: 消息列表
        max_tokens: 最大token数
    
    Returns:
        dict: 响应结果
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=60)
    return response.json()
```

## 使用示例

```python
# 检查端口
if not check_port_available(8000):
    print("端口8000已被占用")

# 启动服务后等待就绪
if wait_for_service_ready(8000, timeout=300):
    # 发送测试请求
    result = send_test_request(8000, [
        {"role": "user", "content": "1+1=?"}
    ])
    print(result["choices"][0]["message"]["content"])
```
