# tutorial

## Colab Cliを使う

```sh
uv tool install google-colab-cli
```

### Scriptを直接実行する

```sh
❯ colab run notebooks/hello.py              
[colab] Creating session 'run-3c1107'...
[colab] Session READY (run-3c1107). Executing notebooks/hello.py...
hello Google Colab!
[colab] Stopping session 'run-3c1107'...
[colab] Session terminated.
```

- セッションを新しく作って、そこで実行
- セッションは自動的に破棄

### Sessionを作ってScriptを実行

```sh
❯ colab new                                 
[colab] Creating session '12368b'...
[colab] Session READY.

❯ colab status
[12368b] m-s-kkb-usw3b1-18v4ekhmr3oyk | Hardware: CPU | Shape: Standard | Variant: DEFAULT | Status: IDLE

❯ colab exec -f notebooks/hello.py 
[colab] Using unique session '12368b'.
hello Google Colab!

❯ colab stop -s 12368b
[colab] Stopping session '12368b'...
[colab] Session terminated.
```

- 明示的にセッションを作り、そこで実行
- セッションはstopしないと維持

### ライブラリなどのインストール

関数をまとめたものをローカルでビルドして、colab環境で使いたいケース

```sh
❯ uv build
Building source distribution...
Building wheel from source distribution...
Successfully built dist/learn_colab_work-0.0.1.tar.gz
Successfully built dist/learn_colab_work-0.0.1-py3-none-any.whl

❯ colab new
[colab] Creating session 'a6bafe'...
[colab] Session READY.

❯ colab upload dist/learn_colab_work-0.0.1-py3-none-any.whl lib/learn_colab_work-0.0.1-py3-none-any.whl
[colab] Using unique session 'a6bafe'.
[colab] Uploaded 'dist/learn_colab_work-0.0.1-py3-none-any.whl' to 'lib/learn_colab_work-0.0.1-py3-none-any.whl

❯ colab install lib/learn_colab_work-0.0.1-py3-none-any.whl
[colab] Using unique session 'a6bafe'.
[colab] Installing packages on a6bafe (preferring uv)...
Installation Complete (via uv)!

❯ colab exec -f notebooks/hello_func.py 
[colab] Using unique session 'a6bafe'.
hello learn_colab_work!

❯ colab stop
[colab] Using unique session 'a6bafe'.
[colab] Stopping session 'a6bafe'...
[colab] Session terminated.
```

- セッションを作った後にファイルをアップロード > インストール

### colab環境に接続する

```sh
❯ colab console
[colab] Using unique session '1f62b7'.

/content# hostname
cde6e69940af
```

- 直接VMに接続する