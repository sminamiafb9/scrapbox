import asyncio


async def task(name):
    # sub task
    print(f"{name}: start")

    # 1秒待つが、Event Loopに処理は返す
    await asyncio.sleep(1)
    print(f"{name}: end")


async def main():
    # sub taskを投げる起点のタスク
    print("main start")

    # taskを投げて終了を待つが、Event Loopに処理は返す
    # taskも中でawaitしており、Event Loopに処理を返すので、
    # task("A")とtask("B")は、個々の完了を待たずに実行されていく
    await asyncio.gather(
        task("A"),
        task("B"),
    )
    print("main end")


if __name__ == "__main__":
    # Event Loopの起動
    # mainを投げて終了を待つが、Event Loopに処理は返す
    # ここは起点のため、mainが終了するまでEvent Loopは生存しプログラム自体は終了しない
    # Event Loop側で await の終了を待ち、終了したTaskを再開して順次処理する
    # その切り替えによって、全体として複数の処理が並行して進む
    asyncio.run(main())
