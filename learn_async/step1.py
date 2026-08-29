import asyncio


async def main():
    print("main start")
    await asyncio.sleep(1)
    print("main end")


if __name__ == "__main__":
    print("script start")
    asyncio.run(main())
    print("script end")
