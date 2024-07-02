import sys
import asyncio
import os
import json

from telethon.tl.types import PeerChannel
from telethon import TelegramClient


api_id = os.environ["API_ID"]
api_hash = os.environ["API_HASH"]


async def main():
    client = TelegramClient("", api_id, api_hash)
    try:
        await client.connect()
    except:
        print("Couldn't connect to Telegram")
        sys.exit(1)

    async for p in client.get_participants(-1001445932373):
        print(p)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
