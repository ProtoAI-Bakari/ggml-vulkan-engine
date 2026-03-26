import aiohttp
import asyncio

async def test():
    async with aiohttp.ClientSession() as s:
        try:
            r = await s.get('http://localhost:8000/health', timeout=aiohttp.ClientTimeout(total=5))
            print('Health OK')
            
            r2 = await s.post(
                'http://localhost:8000/v1/chat/completions',
                json={'model':'mlx-community/GLM-4.7-Flash-8bit','messages':[{'role':'user','content':'2+2'}]},
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            data = await r2.json()
            print('Response:', str(data)[:200])
        except Exception as e:
            print(f'Error: {type(e).__name__}: {str(e)}')

asyncio.run(test())
