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
- **M7** (完了, #15): 1-step教師強制損失をマルチステップ(ロールアウト)損失に
  切り替え、ロールアウト誤差削減の効果を検証する。
- **M8** (完了, #17) / **M9** (完了, #19): M7モデルをM4-M6の制御タスクで再検証。
  ロールアウト誤差の改善が制御タスクの成功率・安定性を保証しないことを確認。
- **M11** (完了, #23): M7が制御タスクで悪化した根本原因を、倒立平衡点での局所線形化
  (ヤコビアン)比較で調査。平衡点そのものでの線形化は真値とほぼ一致しており、実際の
  乖離は平衡点から大きく外れた状態からの回復挙動に現れることを特定。
- **M12** (完了, #25): 初期角速度をスイープしてM11の知見を一般化。1-stepモデルは
  トルク飽和で真値が失敗する領域でも常に成功してしまう楽観バイアスを持つ一方、
  マルチステップモデルの失敗領域は真値の失敗領域の形状とよく一致することを発見。
- **M13** (完了, #27): 現在のサロゲート+PIDを閉ループさせて訪れた状態を真の物理
  モデルでラベル付けするDAgger風のオンポリシーデータ収集で再学習。M12の角速度
  スイープでの成功数が7/15→9/15に改善し、失敗パターンの形状も真値に近づいた。

M1-M9の詳細な要約・横断的な教訓は [LESSONS.md](LESSONS.md) を参照。

## ディレクトリ構成

```
src/
  physics.py               # ODE定義(減衰・トルク項) + RK4シミュレータ
  dataset.py               # 軌道生成・IC サンプリング・分割・制御/ロールアウト窓データセット生成
  model.py                 # NSS(MLP, sin/cos エンコード, トルク入力対応, 微分可能ロールアウト)
  train.py                 # 学習ループ(マルチステップ損失 + カリキュラム学習、curriculum引数で切替可)
  rollout.py               # 自己回帰ロールアウト + 誤差/エネルギー計算
  evaluate.py               # 指標集計・プロット生成(M1-M3, M7のBefore/After比較にも使用)
  control.py                # ランダムシューティングMPC(スイングアップ制御)
  evaluate_control.py       # MPC制御検証・プロット生成(M4)
  pid.py                    # PTP対応PIDコントローラ(重力FF, 真値/サロゲート閉ループ両対応)
  evaluate_pid.py            # 倒立固定目標PID安定化検証・プロット生成(M5)
  evaluate_ptp.py            # PTP制御検証・プロット生成(M6)
  diagnose_linearization.py     # 倒立平衡点でのヤコビアン比較・回復挙動診断(M11)
  diagnose_amplitude_sweep.py   # 初期角速度スイープでの1-step/マルチステップ比較(M12)
  dagger.py                     # DAgger風オンポリシーデータ収集(M13)
  retrain_dagger.py             # DAggerデータ込みの再学習スクリプト(M13)
  evaluate_dagger.py            # DAgger再学習のBefore/After比較・プロット生成(M13)
tests/
  test_physics.py
  test_control.py
  test_pid.py
  test_train.py
  test_diagnose_linearization.py
  test_diagnose_amplitude_sweep.py
  test_dagger.py
```
