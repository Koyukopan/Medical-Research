# 🧠 医療文書検索・評価システム（Medical Research Retrieval）

## 📘 概要

本プロジェクトは、**PMC（PubMed Central）由来の医療文書コーパス**を対象とした  
情報検索（Information Retrieval）実験環境の構築を目的としています。

Retrieval-Augmented Generation（RAG）や有用性評価（AutoMedEval）に発展可能な基盤として、  
検索・リランキング・評価の一連のプロセスを再現可能な形で整備しました。

---

## 🎯 研究目的

- 医療分野において「**関連性**」だけでなく「**臨床的有用性（usefulness）**」を反映した検索技術を目指す  
- 医療従事者・患者が**本当に役立つ文献**にアクセスできる検索スコア設計を検討する  
- 将来的に、**クエリ分解（Decomposed Query）**や**Cross-Encoderによるリランキング**へ発展予定  

---

## 🧩 環境構成

| 項目 | 内容 |
|------|------|
| OS | Ubuntu 24.04 LTS |
| Python | 3.12（仮想環境: `koyuvenv`） |
| OpenSearch | Docker（Port: 9200） |
| 評価ツール | `trec_eval`（NIST公式） |
| 埋め込みモデル | OpenAI `text-embedding-3-small` / `text-embedding-ada-002` / SentenceTransformer(e5-base-v2) |
| データセット | ReCDS（PMC由来の医療文書コーパス） |

---

## 📁 ディレクトリ構成
```bash
Medical-Research/
├── opensearch-beir-benchmarks/   # OpenSearchインデックス作成・評価スクリプト
├── hybrid_rrf_search.py           # 検索システム本体（BM25 + Vector + RRF）
├── README.md
├── .gitignore
└── （ローカルのみ）
├── datasets/                 # 医療文書(PAM)コーパス・クエリ・qrels（GitHub除外）
├── runs/                     # 検索結果（.trec）
└── results/                  # 評価出力
```

> ⚠️ `.gitignore` により、`datasets/`, `runs/`, `results/` はGitHubにアップロードされません。

---

## ⚙️ 環境セットアップ

### 1. 仮想環境の作成

```bash
python3 -m venv koyuvenv
source koyuvenv/bin/activate
pip install -r requirements.txt

```
### 2. OpenSearch の起動
```
cd opensearch-beir-benchmarks
docker compose up -d
```
稼働確認：
```bash
docker ps
# port 9200 が OpenSearch に割り当てられていればOK
```
### 3. コーパスのインデックス作成（例：500件）
```bash
python scripts/index_corpus_chunked.py \
  --dataset_path ../datasets/pmc-recds-mini/corpus_500.jsonl \
  --index_name pmc-recds-mini-500 \
  --opensearch_url http://localhost:9200 \
  --username admin \
  --password 'Uema2@lab' \
  --retrieval_method hybrid \
  --embedding_model text-embedding-3-small \
  --api_token $OPENAI_API_KEY
```

### 4. 検索実行（RRF融合）

hybrid_rrf_search.py は、BM25 + ベクトル検索のRRF融合による
ハイブリッド検索を行います。
```bash
python hybrid_rrf_search.py \
  --input_jsonl ../datasets/pmc-recds-mini/queries.jsonl \
  --index_name pmc-recds-mini-500 \
  --out_json ./runs/run_docs.trec
```

出力（TREC形式）：
```bash
qid  Q0  doc_id  rank  score  run_name
```

### 5. 評価（TREC公式ツール）
```bash
trec_eval ../datasets/pmc-recds-mini/qrels.tsv ./runs/run_docs.trec > ./results/eval_results.txt
```

例：
```bash
map             all   0.4123
ndcg_cut_10     all   0.5632
recall_100      all   0.7811
```


###  作者情報
	•	Author: Koyukopan
	•	Affiliation: 学部4年 卒業研究（情報検索・医療AI）
	•	Keywords: RAG, OpenSearch, BEIR, TREC, Medical NLP

### ライセンス

本リポジトリのコードは MIT License のもとで公開されています。
データセット（PMC由来）は各ソースのライセンスに従ってください。
