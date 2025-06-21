from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import settings


# 定义 Dify 调用时需要在请求头中携带的 API 密钥的名称
API_KEY_NAME = "Authorization" 
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 从环境变量中读取您设置的安全 API 密钥
DIFY_SECRET_KEY = settings.DIFY_API_KEY

async def get_api_key(api_key: str = Security(api_key_header)):
    """
    依赖项函数，用于验证传入的 API 密钥。
    Dify 在调用时，会在请求头中添加 'Authorization: Bearer <your-key>'
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
        )
    
    # Dify 发送的密钥格式为 "Bearer <key>"，我们需要提取 key 的部分
    key_parts = api_key.split(" ")
    if len(key_parts) != 2 or key_parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <key>'",
        )
        
    actual_key = key_parts[1]
    
    if actual_key == DIFY_SECRET_KEY:
        return actual_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API Key",
        )
