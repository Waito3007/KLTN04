import asyncio
import httpx

async def test_stats():
    async with httpx.AsyncClient() as client:
        resp = await client.get('http://localhost:8000/api/github/Waito3007/KLTN04/commit-stats')
        print(f'Status: {resp.status_code}')
        if resp.status_code == 200:
            print('✅ SUCCESS')
            data = resp.json()
            print(f'Statistics: {data}')
        else:
            print('❌ ERROR')
            print(resp.text)

if __name__ == "__main__":
    asyncio.run(test_stats())
