# import hashlib
# from typing import Tuple
# from fastapi import HTTPException, Header
# from loguru import logger as custom_logger
# from sqlalchemy import select
# import hashlib

# from app.api.db.session import AsyncSessionLocal
# from app.api.model.api_key import ApiKey


# class ApiKeyValidator:
#     """Validate API keys and extract user information."""
    
#     @staticmethod
#     def hash_api_key(api_key: str) -> str:
#         """Hash API key for database lookup."""
#         return hashlib.sha256(api_key.encode()).hexdigest()


#     async def validate_api_key(api_key: str) -> Tuple[str, str]:
#         try:
#             if not api_key:
#                 raise HTTPException(
#                     status_code=401,
#                     detail="API key is required"
#                 )
            
#             if not api_key.startswith('bap_'):
#                 raise HTTPException(
#                     status_code=401,
#                     detail="Invalid API key format"
#                 )
            
#             hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
            
#             async with AsyncSessionLocal() as session:
#                 query = select(ApiKey).where(
#                     ApiKey.hashed_key == hashed_key,
#                     ApiKey.is_active == True
#                 )
#                 result = await session.execute(query)
#                 api_key_record = result.scalar_one_or_none()
                
#                 if not api_key_record:
#                     custom_logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
#                     raise HTTPException(
#                         status_code=401,
#                         detail="Invalid API key"
#                     )
                
#                 from datetime import datetime
#                 api_key_record.last_used_at = datetime.utcnow()
#                 await session.commit()
                
#                 user_id = str(api_key_record.user_id)
#                 api_key_id = str(api_key_record.id)
                
#                 custom_logger.info(f"API key validated for user: {user_id}")
#                 return user_id, api_key_id
                
#         except HTTPException:
#             raise
#         except Exception as e:
#             custom_logger.error(f"API key validation failed: {e}")
#             raise HTTPException(
#                 status_code=500,
#                 detail="Authentication service error"
#             )