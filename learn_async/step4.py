import asyncio


async def task(name, seconds) -> str:
    # sub task
    print(f"{name}: start")

    # 1秒待つが、Event Loopに処理は返す
    await asyncio.sleep(seconds)

    print(f"{name}: watch")
    return f"{name}: end"


# async forで処理されるジェネレータの定義
class AsyncGenerator:
    def __init__(self, name, seconds):
        self.name = name
        self.seconds = seconds

    def __aiter__(self):
        return self

    async def __anext__(self):
        """async forで呼び出される1イテレータ分の処理"""

        if not self.seconds:
            # loop処理の終端
            raise StopAsyncIteration

        sec = self.seconds.pop(0)

        # taskの終了を待つが、Event Loopに処理は返す
        res = await task(f"{self.name}-{sec}", sec)

        return res


async def main():
    # sub taskを投げる起点のタスク
    print("main start")

    # 先に一つタスクを生成する
    sub_task = asyncio.create_task(task("sub-task", 2))

    async for res in AsyncGenerator("task", [1, 2, 3]):
        # イテレーション自体は逐次的に進む
        # ただし、次の値を取得する __anext__() は非同期
        # __anext__() 内の await 中はEvent Loopに制御を返す
        # その間、別Taskである sub_task が並行して実行される
        print(res)

    # sub_taskの終了自体はここで待つ
    res = await sub_task
    print(res)

    print("main end")


if __name__ == "__main__":
    # Event Loopの起動
    # mainを投げて終了を待つが、Event Loopに処理は返す
    # ここは起点のため、mainが終了するまでEvent Loopは生存しプログラム自体は終了しない
    # Event Loop側で await の終了を待ち、終了したTaskを再開して順次処理する
    # その切り替えによって、全体として複数の処理が並行して進む
    asyncio.run(main())
