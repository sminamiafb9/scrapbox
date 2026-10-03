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

### Notebookのカーネルとして利用する

`colab new`で作成されたセッションで起動しているjupyterをリモートカーネルとしてvscodeで設定する

```sh
colab new -s learn_colab_work
# パッケージの追加など

# この時点でjupyterがcolabのVMで起動しており、{VMのIPアドレス}:9000で待ち受けている
ssh \
  -o 'ProxyCommand=colab ssh --proxy-mode -s learn_colab_work' \
  -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -N \
  -L 8889:172.28.0.12:9000 \
  root@learn_colab_work
```

- VMのjupyterにsshのポートフォワーディングでアクセスできるようにする
- VScode側でInteractiveモードを起動して、以下の設定を行う
```text
1. カーネル選択
2. 既存のjupyterサーバーを選択
3. 127.0.0.1を選択
4. カーネルを選択
```

### colab環境に接続する

```sh
❯ colab console
[colab] Using unique session '1f62b7'.

/content# hostname
cde6e69940af
```

- 直接VMに接続する

## poeのタスクにする 

```sh
❯ uv run poe colab_deploy                     
Poe => uv build &&
WHEEL=$(find dist -maxdepth 1 -name 'learn_colab_work-*.whl' -print -quit) &&
colab new -s learn_colab_work &&
colab upload "$WHEEL" "lib/$(basename "$WHEEL")" &&
colab install -s learn_colab_work "lib/$(basename "$WHEEL")"
Building source distribution...
Building wheel from source distribution...
Successfully built dist/learn_colab_work-0.0.1.tar.gz
Successfully built dist/learn_colab_work-0.0.1-py3-none-any.whl
[colab] Creating session 'learn_colab_work'...
[colab] Session READY.
[colab] Using unique session 'learn_colab_work'.
[colab] Uploaded 'dist/learn_colab_work-0.0.1-py3-none-any.whl' to 'lib/learn_colab_work-0.0.1-py3-none-any.whl'
[colab] Installing packages on learn_colab_work (preferring uv)...
Installation Complete (via uv)!

❯ uv run poe colab_run notebooks/hello_func.py
      Built learn-colab-work @ file:///Users/minami_shotaro/Works/scrapbox/learn_colab_work                                                      
Uninstalled 1 package in 0.62ms
Installed 1 package in 1ms
Poe => colab exec -s learn_colab_work -f $file
hello learn_colab_work!

❯ colab status                                
[learn_colab_work] m-s-kkb-usc1a2-3dz4enaq811vk | Hardware: CPU | Shape: Standard | Variant: DEFAULT | Status: IDLE
  Last Execution: automation:install at 2026-10-03 10:26:20

❯ uv run poe colab_stop                       
Poe => colab stop -s learn_colab_work
[colab] Stopping session 'learn_colab_work'...
[colab] Session terminated.
```

### カーネル利用のSSHフォワーディングタスク

```sh

```