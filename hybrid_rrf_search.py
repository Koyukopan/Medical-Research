#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 旧R_koyuのconvert_to_doc

"""
Narrative JSONL を読み、サブクエリごとに BM25 とベクトル検索を行い、
RRFで融合して Docランキングを作成。さらに各Docの最良セグメントを
narrative埋め込みとのコサイン類似度で選び、Segmentランキングを出力する。
OpenAIキー不要（ローカル e5-base-v2）。
"""

import json, time
from pathlib import Path
from typing import List, Tuple
from opensearchpy import OpenSearch
from opensearchpy.exceptions import ConnectionError as OSConnErr
from sentence_transformers import SentenceTransformer

# ========= 設定 =========
OS_HOST = "http://localhost:9200"  # HTTPSなら https:// に変更し verify を合わせる
OS_USER = "admin"
OS_PASS = "admin"

DOC_INDEX = "msmarco_v2.1_doc"             # ドキュメントの index（doc_id, text, embedding）
SEG_INDEX = "msmarco_v2.1_doc_segmented"   # セグメントの index（doc_id, seg_id, embedding） 無ければ None
USE_SEG_INDEX = True                        # セグメントindexが無いなら False に

INPUT_JSONL = "/home/ubuntu/dev/repos/simple_msmacro_insert/query_decomp_trec2025_expanded.jsonl"

OUT_DOC_TREC = "./run_docs.trec"
OUT_SEG_TREC = "./run_segments.trec"
RUN_NAME_DOC = "hybrid_rrf_doc"
RUN_NAME_SEG = "hybrid_rrf_seg"

TOPK_BM25 = 200
TOPK_VEC  = 200
TOPK_SUBQ = 200         # サブクエリ内のRRF後に残す件数
TOPK_FINAL_DOC = 100    # 最終Doc件数
RRF_K_INNER = 60        # BM25×Vec 融合
RRF_K_OUTER = 60        # サブクエリ間 融合
SIM_THRESHOLD = 0.65    # セグメント採択しきい値（要調整）

# ========= クライアント =========
def os_client():
    return OpenSearch(
        hosts=[OS_HOST],
        http_auth=(OS_USER, OS_PASS),
        timeout=120,
        retry_on_timeout=True,
        max_retries=5,
        http_compress=True,
        # HTTPS/自己署名を使う場合は下を有効化:
        # use_ssl=True, verify_certs=False, ssl_show_warn=False,
    )

# ========= 埋め込み（e5） =========
_model = SentenceTransformer("intfloat/e5-base-v2")
def embed_query(text: str):
    return _model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()

# ========= RRF =========
def rrf_fuse(rank_lists: List[List[Tuple[str, int]]], rrf_k: int) -> List[Tuple[str, float]]:
    scores = {}
    for lst in rank_lists:
        for doc_id, r in lst:
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + (r+1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def to_rank_pairs(ids_in_order: List[str]) -> List[Tuple[str,int]]:
    return [(did, r) for r, did in enumerate(ids_in_order)]

# ========= 安全検索（リトライ） =========
def safe_search(client, index, body, tries=3, delay=2):
    for i in range(tries):
        try:
            return client.search(index=index, body=body)
        except OSConnErr as e:
            if i == tries - 1:
                raise
            time.sleep(delay * (i + 1))

# ========= BM25 / Vec =========
def search_bm25(client, index, query_text, size):
    body = {"size": size, "query": {"match": {"text": query_text}}, "_source": ["doc_id"]}
    res = safe_search(client, index, body)
    return [h["_source"]["doc_id"] for h in res["hits"]["hits"]]

def search_knn(client, index, query_vector, size):
    body = {"size": size, "query": {"knn": {"embedding": {"vector": query_vector, "k": size}}}, "_source": ["doc_id"]}
    res = safe_search(client, index, body)
    return [h["_source"]["doc_id"] for h in res["hits"]["hits"]]

# ========= セグメント最良1本 =========
def best_segment_for_doc(client, seg_index, doc_id, narrative_vec):
    q = {"size": 1000, "query": {"term": {"doc_id.keyword": doc_id}}, "_source": ["seg_id", "embedding"]}
    res = safe_search(client, seg_index, q)
    best_sid, best_sim = None, -1.0
    for h in res["hits"]["hits"]:
        emb = h["_source"].get("embedding")
        if not emb: 
            continue
        # e5は正規化済み想定 → cos = dot
        cos = sum(a*b for a,b in zip(narrative_vec, emb))
        if cos > best_sim:
            best_sim = cos
            best_sid = h["_source"]["seg_id"]
    return best_sid, best_sim

# ========= サブクエリ拡張（控えめ） =========
def build_subquery_text(sq_text: str, narrative: str, bg_text: str, bg_the: list) -> str:
    # まずは sq + narrative を主、背景は少しだけ（長くしすぎない）
    extra = " ".join(bg_the[:3]) if bg_the else ""
    return f"{sq_text}. {narrative} {bg_text} {extra}".strip()

# ========= メイン =========
def main():
    client = os_client()
    # インデックス健全性
    assert client.indices.exists(index=DOC_INDEX), f"{DOC_INDEX} が存在しません。"

    with open(INPUT_JSONL, "r", encoding="utf-8") as f_in, \
         open(OUT_DOC_TREC, "w", encoding="utf-8") as f_doc, \
         open(OUT_SEG_TREC, "w", encoding="utf-8") as f_seg:

        for line in f_in:
            rec = json.loads(line)
            topic_id = str(rec["id"])
            narrative = rec["narrative"]
            bg_text = rec.get("background_text", "")
            bg_the  = rec.get("background_thesaurus", [])
            subqs   = rec.get("subquery", [])
            if not subqs:
                subqs = [{"text": narrative}]

            # --- サブクエリごとに BM25/Vec → 内側RRF ---
            inner_ranklists = []
            for sq in subqs:
                sq_text = build_subquery_text(sq["text"], narrative, bg_text, bg_the)
                bm25_ids = search_bm25(client, DOC_INDEX, sq_text, TOPK_BM25)
                vec_ids  = search_knn(client, DOC_INDEX, embed_query(sq_text), TOPK_VEC)
                fused    = rrf_fuse([to_rank_pairs(bm25_ids), to_rank_pairs(vec_ids)], RRF_K_INNER)
                inner_ranklists.append([(doc_id, r) for r, (doc_id, _) in enumerate(fused[:TOPK_SUBQ])])

            # --- サブクエリ間 RRF（外側） ---
            fused_docs = rrf_fuse(inner_ranklists, RRF_K_OUTER)
            ranked_docs = fused_docs[:TOPK_FINAL_DOC]  # [(doc_id, score)]

            # ===== Docランキング → TREC出力 =====
            for rank, (doc_id, score) in enumerate(ranked_docs, start=1):
                f_doc.write(f"{topic_id}\tQ0\t{doc_id}\t{rank}\t{score:.6f}\t{RUN_NAME_DOC}\n")

            # ===== セグメント抽出 =====
            if USE_SEG_INDEX and client.indices.exists(index=SEG_INDEX):
                nar_vec = embed_query(narrative)
                seg_items = []
                for doc_id, doc_score in ranked_docs:
                    seg_id, sim = best_segment_for_doc(client, SEG_INDEX, doc_id, nar_vec)
                    if seg_id and sim >= SIM_THRESHOLD:
                        # Docスコアとセグメント類似度の合成（軽めの重み）
                        final = doc_score * 0.7 + sim * 0.3
                        seg_items.append((f"{doc_id}#{seg_id}", final))

                seg_items.sort(key=lambda x: x[1], reverse=True)
                for rank, (docseg, score) in enumerate(seg_items, start=1):
                    f_seg.write(f"{topic_id}\tQ0\t{docseg}\t{rank}\t{score:.6f}\t{RUN_NAME_SEG}\n")

if __name__ == "__main__":
    main()
