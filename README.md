# surrogateRolloutVerification

振り子の物理モデル(ODE)をNeural State Space (NSS) モデルでサロゲート化し、
自己回帰ロールアウト時の誤差蓄積を検証するプロジェクト。

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## マイルストーン

- **M1**: 単振り子(無減衰・無入力)で NSS 状態遷移モデルを学習し、
  ロールアウト誤差蓄積カーブ・エネルギー保存逸脱・位相空間軌道乖離を評価する。
- **M2**: 減衰項を追加し、初期条件分布を倒立域まで拡張する。
- **M3**: トルク入力 `τ` を状態遷移に組み込み、制御データで学習する。
- **M4**: 学習済み NSS を用いた PTP / スイングアップ制御器の検証。

## ディレクトリ構成

```
src/
  physics.py    # ODE定義 + RK4シミュレータ
  dataset.py    # 軌道生成・IC サンプリング・分割
  model.py      # NSS(MLP, sin/cos エンコード)
  train.py      # 学習ループ
  rollout.py    # 自己回帰ロールアウト + 誤差/エネルギー計算
  evaluate.py   # 指標集計・プロット生成
tests/
  test_physics.py
```
