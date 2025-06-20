import asyncio
from db.database import database
from db.models.repositories import repositories
from sqlalchemy import select

async def check_repos():
    await database.connect()
    repos = await database.fetch_all(select(repositories))
    print(f'Found {len(repos)} repositories:')
    for repo in repos:
        print(f'  - {repo.full_name}')
    await database.disconnect()

if __name__ == "__main__":
    asyncio.run(check_repos())
