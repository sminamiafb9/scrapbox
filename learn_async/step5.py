import asyncio


async def task(name, seconds) -> str:
    # sub task
    print(f"{name}: start")

    # 1秒待つが、Event Loopに処理は返す
    await asyncio.sleep(seconds)

    print(f"{name}: watch")
    return f"{name}: end"


class AsyncResource:
    async def __aenter__(self):
        print("enter start")
        await asyncio.sleep(1)
        print("enter end")

    async def __aexit__(self, exc_type, exc, tb):
        print("exit start")
        await asyncio.sleep(1)
        print("exit end")


async def main():
    print("main start")

    # async with は、コンテキストの開始処理 __aenter__() と終了処理 __aexit__() を非同期に実行
    async with AsyncResource():
        print("inside")

    print("main end")


if __name__ == "__main__":
    asyncio.run(main())
