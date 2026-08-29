# learn async

- step1
  - 最小実行コード
  - 1秒待つ
- step2
  - 1秒待つタスクを非同期に2つ実行
- step3
  - 1, 2, 3秒待つタスクを非同期に実行
- step4
  - async forのサンプル
  - ジェネレータの中(生成処理)でawaitで非同期処理が走る
- step5
  - async withのサンプル
  - コンテキスト処理(enter, exit)でawaitで非同期処理が走る

## step1 ~ 3の構造図

```mermaid
sequenceDiagram
    participant P as Program
    participant E as EventLoop
    participant M as MainTask
    participant C as Coroutine

    P->>E: asyncio.run(main())
    E->>M: mainを実行

    M->>C: coroutineを実行
    C->>E: await

    Note over E: 待機中は別Taskを実行できる

    E->>C: Coroutineを再開
    C-->>M: 結果を返す

    M->>E: main完了
    E->>P: asyncio.run()終了
```

## step4 (async for)

```mermaid
sequenceDiagram
    participant M as Main
    participant E as EventLoop
    participant I as AsyncIterator

    M->>E: await anext(iterator)
    E->>I: __anext__()
    I->>E: await中
    Note over E: 別Taskを実行できる
    I-->>M: 次の値
    M->>M: ループ本体を実行

    M->>E: await anext(iterator)
    E->>I: __anext__()
    I->>E: await中
    Note over E: 別Taskを実行できる
    I-->>M: 次の値
    M->>M: ループ本体を実行

    M->>E: await anext(iterator)
    E->>I: __anext__()
    I-->>M: StopAsyncIteration
    M->>M: async for終了
```

## step5 (async with)

```mermaid
sequenceDiagram
    participant M as Main
    participant E as EventLoop
    participant C as AsyncContext

    M->>E: await __aenter__()
    E->>C: __aenter__()
    C->>E: await中
    Note over E: 別Taskを実行できる
    C-->>M: resource
    M->>M: with本体を実行

    M->>E: await __aexit__()
    E->>C: __aexit__()
    C->>E: await中
    Note over E: 別Taskを実行できる
    C-->>M: 終了処理完了

    M->>M: async with終了
```

## まとめ

- asyncio.run
  - Event Loopの実行
- async def
  - 関数を「途中で await してEvent Loopに制御を返せるCoroutine」にする
- async for
  - 次の値を取得する処理（\_\_anext\_\_）で await してEvent Loopに制御を返せる
- async with
  - コンテキストの開始（\_\_aenter\_\_）・終了（\_\_aexit\_\_）で await してEvent Loopに制御を返せる

```mermaid
flowchart TD
    P[Main Script]
    R[asyncio.run]
    E[Event Loop]
    M[main Task]

    P -->|asyncio.run| R
    R -->|起動| E
    E -->|実行| M

    M --> C[Coroutine]
    M --> I[AsyncIterator]
    M --> X[AsyncContextManager]

    C --> W[待機]
    I --> W
    X --> W

    W -->|制御を返す| E
```