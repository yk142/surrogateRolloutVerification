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

- **M1** (完了, #1): 単振り子(無減衰・無入力)で NSS 状態遷移モデルを学習し、
  ロールアウト誤差蓄積カーブ・エネルギー保存逸脱・位相空間軌道乖離を評価する。
- **M2** (完了, #3): 減衰項を追加し、初期条件分布を倒立域・完全回転域まで拡張する。
- **M3** (完了, #5): トルク入力 `τ` を状態遷移に組み込み、制御データで学習する。
- **M4** (完了, #7): 学習済み NSS を用いたスイングアップ/PTP制御(ランダムシューティングMPC)の検証。
- **M5** (完了, #9): 同一PIDゲインを真の物理モデル/NSSサロゲートに適用し、倒立近傍の
  局所安定化ダイナミクスの一致度を検証する。
- **M6** (完了, #11): PIDを目標角度可変(重力フィードフォワード込み)に一般化し、
  倒立近傍でのPTP(点対点)制御を真の物理モデル/NSSサロゲートで比較検証する。

## ディレクトリ構成

```
src/
  physics.py         # ODE定義(減衰・トルク項) + RK4シミュレータ
  dataset.py         # 軌道生成・IC サンプリング・分割・制御データセット生成
  model.py           # NSS(MLP, sin/cos エンコード, トルク入力対応)
  train.py           # 学習ループ
  rollout.py         # 自己回帰ロールアウト + 誤差/エネルギー計算
  evaluate.py         # 指標集計・プロット生成(M1-M3)
  control.py          # ランダムシューティングMPC(スイングアップ制御)
  evaluate_control.py # MPC制御検証・プロット生成(M4)
  pid.py              # PTP対応PIDコントローラ(重力FF, 真値/サロゲート閉ループ両対応)
  evaluate_pid.py      # 倒立固定目標PID安定化検証・プロット生成(M5)
  evaluate_ptp.py      # PTP制御検証・プロット生成(M6)
tests/
  test_physics.py
  test_control.py
  test_pid.py
```
