import asyncio
import cognee

async def reset():
    await cognee.forget(everything=True)
    print("Cognee reset complete.")

asyncio.run(reset())