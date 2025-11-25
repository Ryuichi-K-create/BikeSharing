# Bike Sharing Demand Prediction

## 概要 (Overview)
本プロジェクトは、米国ワシントンD.C.の自転車シェアリングシステム「Capital Bikeshare」の過去の利用履歴と気象データを用いて、将来の自転車レンタル需要を予測する機械学習モデルを構築・評価したものです。

線形モデルである**Ridge回帰**と、非線形モデルである**多層パーセプトロン（MLP）**を実装し、その予測精度を比較検証しました。結果として、MLPが複雑な需要パターンを捉え、ベースラインと比較して大幅な精度向上（RMSLE 約48%改善）を達成しました。

本リポジトリには、データ前処理からモデル学習、評価、そして結果をまとめたLaTeXレポートまでの全ソースコードが含まれています。

## 使用技術 (Technologies)
- **Language**: Python 3.11
- **ML Frameworks**: PyTorch (for MLP), Scikit-learn (for Ridge, Preprocessing)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Documentation**: LaTeX (uplatex, dvipdfmx)

## 手法 (Methodology)

### 1. データ前処理と特徴量エンジニアリング
- **対数変換**: ターゲット変数（レンタル数）の分布が右に裾を引く形状であるため、`log(1+y)` 変換を適用して正規分布に近づけました。
- **One-Hot Encoding**: 時間（hour）、月（month）、曜日（weekday）などの周期的・カテゴリカルな変数をOne-Hotベクトル化し、線形モデルでも非線形なトレンド（例：朝夕のラッシュアワー）を捉えやすくしました。
- **スケーリング**: 気温や湿度などの連続値変数を `StandardScaler` で正規化しました。
- **時系列分割**: 未来のデータが学習に混入する「リーク」を防ぐため、ランダムシャッフルを行わず、時系列順にデータを分割（Train: 60%, Val: 20%, Test: 20%）しました。

### 2. 採用モデル
- **Ridge回帰 (Baseline)**: 多重共線性を防ぐL2正則化項を持つ線形回帰モデル。
- **多層パーセプトロン (MLP)**: PyTorchを用いて実装した深層学習モデル。
    - 構造: 入力層(60次元) -> 隠れ層(50) -> 隠れ層(50) -> 出力層(1)
    - 活性化関数: Tanh
    - 最適化: Adam
    - 正則化: Dropout, Weight Decay, Early Stopping

## 実験結果 (Results)
評価指標には **RMSLE (Root Mean Squared Logarithmic Error)** を使用しました。

| Model | Test RMSLE | 備考 |
|-------|------------|------|
| Ridge Regression | 0.6210 | ベースライン |
| **MLP (PyTorch)** | **0.3196** | **Best Model** |

MLPはRidge回帰と比較して、誤差を大幅に削減することに成功しました。特に、日々の需要のピークや細かい変動において、MLPの方が実測値への追従性が高いことが確認されました。

## ディレクトリ構成 (Project Structure)
```
BikeSharing/
├── main.py          # メインスクリプト（データ読み込み、前処理、学習、評価、可視化）
├── src/
│   ├── __init__.py
│   ├── features.py  # 特徴量エンジニアリング（One-Hot Encoding等）の実装
│   └── models.py    # モデル定義（RidgeWrapper, MLP）の実装
├── data/            # データセットディレクトリ
│   ├── train.csv    # 学習用データ
│   └── test.csv     # テスト用データ
├── output/          # 出力ディレクトリ
│   ├── loss_curve.png            # 学習曲線
│   ├── prediction_comparison.png # 予測値の散布図比較
│   ├── timeseries_mlp.png        # MLPの時系列予測プロット
│   ├── timeseries_ridge.png      # Ridgeの時系列予測プロット
│   └── experiment_log.csv        # 実験ログ
├── report.tex       # 研究レポートのLaTeXソース
└── report.pdf       # 生成されたレポートPDF
```

## 実行方法 (How to Run)

### 1. 環境構築
必要なライブラリをインストールします。
```bash
pip install pandas numpy scikit-learn matplotlib torch
```

### 2. モデルの学習と評価
以下のコマンドで `main.py` を実行すると、前処理、学習、評価、プロット生成が一括で行われます。
```bash
python main.py
```
実行後、コンソールにRMSLEスコアが表示され、`output/` ディレクトリにグラフ画像が保存されます。

### 3. レポートの生成
LaTeX環境（TeX Liveなど）が必要です。
```bash
uplatex report.tex
dvipdfmx report.dvi
```
これにより `report.pdf` が生成されます。

## 今後の課題 (Future Work)
- **時系列モデルの導入**: LSTMやGRUなどのリカレントニューラルネットワーク（RNN）を用いて、時間的な順序関係をより直接的にモデル化する。
- **外部データの活用**: 降水量やイベント情報など、需要に影響を与える外部データを追加して精度向上を図る。

## Author
[Your Name / Portfolio URL]
