---
layout: post
title: "Shipping RAG: The Quest for Quality"
date: 2026-06-03
categories: [rag, ai, engineering]
excerpt: "After the initial build: quality audits, hybrid retrieval, contextual embeddings, and the cost of measuring what you build."
---

*Companion to [Part 1: What I Learned Building a RAG System on Real, Messy Data](https://snijsure-personal.github.io/2026/05/17/rag-system-real-messy-data/).*

---

## Where Part 1 Left Off

Part 1 was about getting [PermitIQ](https://www.permit-iq.com/) to the point where it returned plausible answers across 60+ city municipal codes. Scraping. Chunking. Embedding. The data pipeline. The standard [RAG (Retrieval-Augmented Generation)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) playbook with some changes I made along the way.

By the end of that article the system worked. I had deployed it, the URL resolved, and "build an [ADU (Accessory Dwelling Unit)](https://en.wikipedia.org/wiki/Accessory_dwelling_unit) in Berkeley" came back with a sensible answer and cited sections.

That was the easy part.

This post is what happened after I started actually using it, what broke, what I changed, what I measured, and how much it cost me to find out.

---

## What I Learned After Shipping

I had quietly assumed that if Oakland worked well, the other 59 cities would be in the same ballpark. They were not. The first time I ran a single question against every city in the system, more than a third of them returned garbage. That kicked off everything that follows.

I'll walk through it in the order it happened: the audit that revealed the problem, three bugs I had to fix, three improvements I shipped, and one measurement framework I built so I could stop guessing.

---

## 1. The Quality Audit: One Question, 35 Cities

I wrote a short async script that hit `/api/chat` for every live city in the system with the same question:

> *"What permits do I need for a kitchen remodel?"*

Then I dumped the answers into a single file and read all of them.

**Fourteen of thirty-five cities returned garbage or nothing.** Root causes, in roughly the order I uncovered them:

- **Empty Indexes:** Several cities had been flipped to `live` status before their scrape job had actually run successfully. 
- **Bot Protection:** Some cities were behind bot-protection systems. The scraper got a 200 response with an "Access Denied, please complete the captcha" page in the body. The system happily indexed the rejection notice as if it were code.
- **Stale Seeds:** City websites redesigned, URLs 404'd, and the embedding model dutifully encoded "Page Not Found" as a vector.

The takeaway: **you don't know your RAG quality until you test it systematically.** A quality gate script should be a deployment step, not an afterthought. 

---

## 2. The HNSW Filtered Query Bug

Most cities had not just thin data; they were returning zero results for *every* query. I assumed I had broken the retrieval pipeline. I had not. I had broken the index.

I had recently migrated the database from per-city schemas into one shared `embeddings` table with a `city` column. The advantage was one [HNSW (Hierarchical Navigable Small World)](https://en.wikipedia.org/wiki/Hierarchical_navigable_small_world) index instead of 60. The bug was that the HNSW index now spanned all cities globally.

The query I was running looked like this:

```sql
SELECT ... FROM embeddings
WHERE city = 'houston'
ORDER BY embedding <=> $1::vector
LIMIT 12;
```

What HNSW actually does on that query: it walks the graph and returns the 12 globally nearest vectors. Then PostgreSQL applies the `WHERE city = 'houston'` filter to those 12 rows. The global nearest neighbors are dominated by Denver (178k rows) and Austin (104k rows), both of which have very dense embedding spaces. After the filter, you get zero Houston rows. The user sees "No results found."

The fix was a one-line GUC (Grand Unified Configuration parameter) introduced in `pgvector` 0.8.0:

```sql
SET hnsw.iterative_scan = relaxed_order;
SELECT ... FROM embeddings WHERE city = $1
ORDER BY embedding <=> $2 LIMIT 12;
```

`iterative_scan = relaxed_order` tells the planner to keep expanding the HNSW search until it accumulates enough rows that *also* satisfy the `WHERE` clause. 

There was a second, smaller bug hiding behind the first: `SET` and `SELECT` need to run on the same database connection. My code was using two separate `pool.query()` calls, which were grabbing different pooled clients. The `SET` was effectively a no-op for the subsequent `SELECT`. Switching to a dedicated `pool.connect()` for the pair fixed it.

---

## 3. The html2text Code-Fence Trap

I was using `html2text` to convert scraped HTML into Markdown. Most HTML converts cleanly; `<table>` does not. What html2text does with tables is wrap them in `<pre>` blocks. The chat [UI (User Interface)](https://en.wikipedia.org/wiki/User_interface) then renders them inside triple-backtick code fences. Unreadable.

The fix was a marker-substitution pattern. Before html2text runs, I walk the soup, convert tables to pipe-formatted markdown, and replace them with a placeholder string (`TABLEPLACEHOLDERiEND`). After html2text returns, I swap the placeholders back for the real Markdown tables.

**Lossy conversions are easier to fight upstream than downstream.**

---

## 4. The System Prompt: Banning the Apology

Even cities with good data were producing answers like: *"Unfortunately, the retrieved sections don't specifically address..."*

The model's default is to hedge when uncertain. You have to explicitly ban the behavior.
- *"Do NOT open with disclaimers, apologies, or 'unfortunately'. Lead directly with the answer."*
- *"First share everything the retrieved sections DO say, then add ONE brief closing note if something is genuinely missing."*

The texture of the answers changed immediately. Leading with substance matters.

---

## 5. "What Can I Ask?": Coverage Before the First Question

A user lands on a thin-data city, asks a question, gets a bad answer, and loses trust in the entire system. 

The fix: surface coverage information upfront. `GET /api/coverage/[cityId]` queries the database for chunk counts and a random sample of breadcrumb paths. A small [LLM (Large Language Model)](https://en.wikipedia.org/wiki/Large_language_model) then summarizes these into 3–4 plain-English sentences:

> *"Oakland's indexed sections cover building permit requirements extensively, including ADU rules, electrical/plumbing/mechanical permits... Business license requirements are not covered."*

The honest framing of what the system *doesn't* cover pre-empts the trust-killer.

---

## 6. Hybrid Search: BM25 + Dense Vectors + RRF

Pure vector search fails on exact-match queries like "Section 420.6" or "Title 17". Semantic search is great for intent; it's terrible for legal citations.

The fix is hybrid search: 
- **Dense pass**: HNSW cosine similarity.
- **Sparse pass**: PostgreSQL full-text search (BM25 ranking).
- **Fusion**: [Reciprocal Rank Fusion (RRF)](https://en.wikipedia.org/wiki/Rank_fusion), `score = Σ 1/(rank + 60)`. 

Rank fusion sidesteps the normalization problem between cosine distances and [BM25 (Best Match 25)](https://en.wikipedia.org/wiki/Okapi_BM25) scores. I blend them at a **0.7/0.3** ratio—dense still dominates, but BM25 gets to "vote" for exact matches.

The database side was handled with a [GIN (Generalized Inverted Index)](https://en.wikipedia.org/wiki/Generalized_Inverted_Index) built `CONCURRENTLY` to avoid table locks:
```sql
CREATE INDEX CONCURRENTLY embeddings_fts_idx
ON embeddings USING gin (to_tsvector('english', coalesce(text, '')));
```

---

## 7. Contextual Retrieval: Anthropic's Technique, at Scale

Chunks in isolation lack context. A chunk saying *"Maximum height is 18 feet"* could be about fences or ADUs. [Anthropic's contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) solves this by prepending a one-sentence summary of the parent document to each chunk before embedding.

I ran this across 600,000 chunks using a Cloud Run job. Cost: ~$57 in Gemini Flash Lite calls. Lift: **+10% faithfulness** and **+5% context precision** on Oakland. 

---

## 8. RAGAS: Building a Real Evaluation Loop

Building a RAG system without evaluation is equivalent to refactoring code without tests. I built a RAGAS-style evaluator measuring **Faithfulness** (no hallucinations) and **Context Precision** (retrieval quality).

### The "Thinking Token" Trap
I used Gemini 2.5 Pro as the judge. Initially, I set `max_output_tokens=8` for YES/NO calls. But Gemini's internal "thinking" tokens consumed the budget before it could output "YES". The fix was bumping the ceiling to 256.

### The 5-City Results

| City | Faithfulness | Context Precision |
|------|--------------|-------------------|
| Oakland       | 0.347 | 0.513 |
| Berkeley      | 0.417 | 0.423 |
| San Francisco | 0.487 | 0.622 |
| Irvine        | 0.572 | 0.099 |
| Denver        | 0.290 | 0.319 |
| **Average**   | **0.423** | **0.395** |

The takeaway? **Retrieval is the variable, generation is the constant.** Faithfulness is stable; Context Precision is all over the map. Irvine's 0.10 precision is a retrieval emergency I never would have seen without the numbers.

---

## 9. Claude → Gemini in the Live App

Economics forced a migration from Claude Sonnet to Gemini 2.5 Flash. Cost dropped from **$0.04 to $0.005 per chat turn**—an 8x reduction.

However, Gemini surfaced **Citation Stacking**: citing 12 identical chunks for one rule.
*"Maximum height is 18 feet [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]."*

The fix:
1. **Deduplicate** chunks by exact text match before they reach the model.
2. **Aggressive Prompting**: *"Never write [1][2][3]…[12]. That is noise."*

---

## What's Next
- **Temporal Versioning:** When was this section last updated?
- **Entity Extraction:** Turning ordinance numbers and fee amounts into metadata filters.
- **Continuous Eval in [CI (Continuous Integration)](https://en.wikipedia.org/wiki/Continuous_integration):** Catching regressions at [PR (Pull Request)](https://en.wikipedia.org/wiki/Pull_request) time.

---

## Musings: The Engineering is in the Wrappers

In Part 1, I noted that the LLM is rarely the bottleneck. Phase 2 proved it. Everything I shipped—HNSW fixes, hybrid search, contextual embeddings, RAGAS—was a data-or-systems problem. 

There is a temptation to attribute AI quality to the model itself. But the model is the most stable component. The real engineering happens in the "wrappers": the chunker, the retriever, the database, and the evaluation loop. 

Phase 1 was about proving it could be done. Phase 2 was about proving it could be *engineered*. 

---

## About This Project
PermitIQ was built on my own time. Total spend: $200–250, mostly on embeddings and evaluation. Storage and serving costs remain negligible. 

Thanks for reading.