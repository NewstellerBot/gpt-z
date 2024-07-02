import sys
import os
import asyncio
import json

from telethon.tl.types import PeerChannel
from telethon import TelegramClient

api_id = os.environ["API_ID"]
api_hash = os.environ["API_HASH"]


async def main():
    client = TelegramClient("cs156", api_id, api_hash)
    try:
        await client.connect()
    except:
        print("Couldn't connect to Telegram")
        sys.exit(1)

    data = []
    ix = 0

    async for message in client.iter_messages(-1001445932373):
        m = {}
        try:
            if type(message.from_id) == PeerChannel:
                m["user_id"] = message.from_id.channel_id
            else:
                m["user_id"] = message.from_id.user_id
        except:
            m["user_id"] = "None"

        m["content"] = message.message
        m["timestamp"] = message.date.timestamp()
        m["id"] = message.id

        data.append(m)
        ix += 1

        if (ix + 1) % 1_000 == 0:
            print(ix)

    with open("data.json", "w") as f:
        f.write(json.dumps(data))


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
