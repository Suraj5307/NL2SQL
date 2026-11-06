---
title: NL2SQL Insight Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
hardware: cpu-basic
tags:
  - nlp
  - sql
  - rag
  - gradio
  - llm
  - sentence-transformers
  - groq
  - ecommerce
  - text2sql
---

# 🤖 NL → SQL Insight Chatbot

Ask business questions in plain English. The system finds the best pre-written SQL query using **semantic search**, runs it against a SQLite database, and uses a **Groq LLM** to explain the result in plain English.

> **Trainer-friendly version** — every step is documented so learners can follow the full pipeline.

---

## 🗺️ Architecture

```
ecommerce.csv  ──►  SQLite DB
                        ▲
10 SQL queries ─────────┤  (run here)
       │
       └─ embed descriptions ─► vectors

User question ─► embed ─► cosine similarity ─► best query ─► run SQL ─► result
                                                                            │
                                               Groq LLM ◄── question + result
                                                   │
                                           plain-English answer
```

**Key design choice:** The LLM never writes SQL — it only reads the SQL result and explains it. This prevents hallucinated numbers.

---

## 🚀 Quick Start (Google Colab)

1. Open `NL2SQL_Simple_Dashboard.ipynb` in Google Colab
2. Upload your CSV file when prompted (Step 2)
3. Paste your free [Groq API key](https://console.groq.com/keys) in Step 1
4. Run all cells top to bottom
5. Click the Gradio public link to open the chat UI

---

## 🛠️ Technology Stack

| Component | Library | Purpose |
|---|---|---|
| Data store | SQLite (`sqlite3`) | Zero-install file database |
| Data loading | Pandas `read_csv` → `to_sql` | CSV → SQL table in one line |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 22 MB model, runs on CPU |
| Similarity search | NumPy dot product | Cosine similarity on normalised vectors |
| LLM | Groq `llama-3.1-8b-instant` | Free tier, very fast |
| Charts | Plotly Express | Auto-detects chart type |
| UI | Gradio 5.x | Web app in ~30 lines |

---

## 📦 Installation

```bash
pip install sentence-transformers groq pandas numpy plotly "gradio>=5.0,<6.0"
```

---

## 📁 Repository Structure

```
├── NL2SQL_Simple_Dashboard.ipynb   # Main trainer notebook (8 steps + 2 bonus cells)
├── app.py                          # Gradio app entry point (for HF Spaces)
├── ecommerce_sales_34500.csv       # Sample dataset
└── README.md                       # This file
```

---

## 📓 Notebook Steps

| Step | What it does |
|---|---|
| Step 0 | Install libraries |
| Step 1 | Configuration (API key, model, paths) |
| Step 2 | Load CSV into SQLite |
| Step 3 | Define 10 vetted SQL queries with descriptions |
| Step 4 | Embed descriptions for semantic search |
| Step 5 | SQL execution helper (read-only) |
| Step 6 | Groq LLM explanation helper |
| Step 7 | Full pipeline function (`answer_question`) |
| Step 8 | Gradio chat UI |
| Bonus | Static dashboard (6 Plotly charts) |
| Bonus | Retrieval accuracy test (20 cases) |

---

## 💬 Example Questions to Try

- *"Which product category generates the most revenue?"*
- *"Who are our top 5 customers by total spend?"*
- *"Show me the monthly sales trend"*
- *"Which region has the highest return rate?"*
- *"What payment methods do customers prefer?"*
- *"Which big orders got heavy discounts?"*

---

## ⚙️ Configuration

All settings live in **Step 1** (one cell):

```python
GROQ_API_KEY  = "your_key_here"          # from console.groq.com — free tier
MODEL_NAME    = "llama-3.1-8b-instant"   # or: llama-3.3-70b-versatile
CSV_PATH      = "/content/your_data.csv"
MIN_SIMILARITY = 0.35                    # reject matches below this score
```

---

## 🔑 Getting a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com/keys)
2. Sign up (free, no credit card needed)
3. Click **Create API Key**
4. Copy the key and paste it into Step 1 of the notebook

---

## 🧠 How Semantic Search Works

1. Each SQL query has a plain-English `description` (and 2 example phrasings)
2. At setup time, all descriptions are converted to 384-dimension vectors using `all-MiniLM-L6-v2`
3. When a user types a question, it is embedded using the same model
4. Cosine similarity (dot product of normalised vectors) ranks all queries
5. The highest-scoring query is selected — if the score is below `MIN_SIMILARITY` (0.35), the system says it cannot answer rather than guessing

---

## 🛡️ Why the LLM Never Writes SQL

Sending user input to an LLM to generate SQL creates two problems:

1. **Hallucination** — the LLM may invent plausible-looking but wrong SQL
2. **SQL injection** — a malicious user could extract or delete data

This system avoids both by keeping a **fixed query library**. The LLM's only job is to read the SQL result (sent as CSV) and explain it in plain English. The system prompt explicitly says: *"use ONLY the numbers from the data."*

---

## 🧪 Retrieval Accuracy

The notebook includes a 20-question accuracy test. With the default descriptions and example phrasings, the system achieves **100% top-1 accuracy** on the included test set.

To test with your own questions, add rows to `TEST_CASES` in the Bonus cell.

---

## ❗ Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Data incompatible with messages format` | Missing `value=[]` on `gr.Chatbot` | Add `value=[]` to `gr.Chatbot(type="messages", value=[])` |
| `groq.AuthenticationError` | Invalid API key | Get a fresh key from [console.groq.com](https://console.groq.com/keys) |
| `sqlite3.OperationalError: no such table` | Step 2 not run yet | Run the CSV → SQLite cell before the SQL steps |
| `ModuleNotFoundError` | pip install not run | Run Step 0 and restart the runtime |
| Score too low, no answer | Question phrasing too different | Add more `examples` to the matching query in Step 3 |

---

## 💡 Suggested Improvements

### Beginner
- Add 2–3 more SQL queries in Step 3
- Adjust `MIN_SIMILARITY` and observe which questions start failing
- Change the LLM system prompt to reply in a different language

### Intermediate
- Add a feedback button that logs thumbs up/down to a CSV
- Route questions to multiple CSV sources based on keywords
- Display query confidence score visually in the UI

### Advanced
- Replace NumPy dot product with FAISS for >1 000 queries
- Add multi-turn memory (pass last 3 exchanges to the LLM)
- Deploy as a FastAPI + React app for production

---

## 📄 License

MIT — free to use, modify, and distribute.

---

## 🙋 For Trainers

This notebook is designed to be taught step by step. Each cell starts with a **"Why?"** markdown block explaining the concept before the code. Suggested teaching order:

1. Run Steps 0–2 and explain the CSV → SQLite pattern
2. Walk through Step 3 (query library design) — this is the most important design decision
3. Live-demo Steps 4–5 to show semantic search in action
4. Show Step 6 and explain grounding / hallucination prevention
5. Run Step 7 and ask the class to suggest questions
6. Run the Bonus accuracy test to show robustness

**Discussion questions:**
1. Why does the LLM receive only the SQL result, not the full database?
2. What happens if two queries have very similar descriptions?
3. Why is a similarity threshold important for a production system?
4. What are the privacy implications of sending business data to an external LLM API?
