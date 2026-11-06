"""
NL → SQL Insight Chatbot
Hugging Face Spaces entry point

Required Space Secrets (Settings → Variables and secrets):
  GROQ_API_KEY  — free key from https://console.groq.com/keys

Required files in the Space repo:
  app.py                        (this file)
  ecommerce_sales_34500.csv     (your dataset)
  README.md                     (Space card)
"""

import os
import re
import time
import sqlite3
from pathlib import Path

# Tell HF Spaces this is a CPU-only app — prevents ZeroGPU redirect
try:
    import spaces  # installed by HF build; fine to ignore if missing locally
    @spaces.GPU(duration=0)
    def _noop(): pass
except Exception:
    pass

import numpy as np
import pandas as pd
import gradio as gr
from groq import Groq
from sentence_transformers import SentenceTransformer


# ── Configuration ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # set as a Space Secret
MODEL_NAME   = "llama3-8b-8192"
CSV_PATH     = "ecommerce_sales_34500.csv"
DB_PATH      = "ecommerce.db"
TABLE_NAME   = "ecommerce_data"
MIN_SIMILARITY = 0.35

# ── Load CSV → SQLite (once at startup) ──────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
conn.commit()
conn.close()
print(f"✅ {len(df):,} rows loaded into SQLite")

# ── SQL query library ─────────────────────────────────────────────────────────
QUERIES = [
    {
        "query_id": "q1",
        "description": "Total sales revenue, total profit, and total completed orders (excluding returned items)",
        "examples": ["How much money did we make overall?", "What is our overall revenue and profit?"],
        "sql": "SELECT SUM(total_amount) AS total_revenue, SUM(profit_margin) AS total_profit, COUNT(DISTINCT order_id) AS total_completed_orders FROM ecommerce_data WHERE returned = 'No';"
    },
    {
        "query_id": "q2",
        "description": "Top 5 highest spending customers by total revenue",
        "examples": ["Who are our best customers?", "Which clients spent the most money?"],
        "sql": "SELECT customer_id, SUM(total_amount) AS total_spent, COUNT(order_id) AS num_orders FROM ecommerce_data WHERE returned = 'No' GROUP BY customer_id ORDER BY total_spent DESC LIMIT 5;"
    },
    {
        "query_id": "q3",
        "description": "Revenue and profit by product category",
        "examples": ["Which category sells the most?", "What categories earn the most profit?"],
        "sql": "SELECT category, SUM(total_amount) AS total_revenue, SUM(profit_margin) AS total_profit, AVG(profit_margin) AS avg_profit_per_order FROM ecommerce_data WHERE returned = 'No' GROUP BY category ORDER BY total_revenue DESC;"
    },
    {
        "query_id": "q4",
        "description": "Sales and return count by region",
        "examples": ["How do sales differ by region?", "Which area has the most revenue and returns?"],
        "sql": "SELECT region, COUNT(order_id) AS total_orders, SUM(total_amount) AS total_revenue, SUM(CASE WHEN returned = 'Yes' THEN 1 ELSE 0 END) AS returned_orders FROM ecommerce_data GROUP BY region ORDER BY total_revenue DESC;"
    },
    {
        "query_id": "q5",
        "description": "Most used payment methods and their total transaction value",
        "examples": ["What payment options do customers prefer?", "Is UPI or credit card used more?"],
        "sql": "SELECT payment_method, COUNT(order_id) AS transaction_count, SUM(total_amount) AS total_revenue FROM ecommerce_data GROUP BY payment_method ORDER BY transaction_count DESC;"
    },
    {
        "query_id": "q6",
        "description": "Average delivery time in days and shipping cost by region",
        "examples": ["How long does delivery take in each region?", "Which region has the slowest delivery?"],
        "sql": "SELECT region, AVG(delivery_time_days) AS avg_delivery_days, AVG(shipping_cost) AS avg_shipping_cost FROM ecommerce_data GROUP BY region ORDER BY avg_delivery_days DESC;"
    },
    {
        "query_id": "q7",
        "description": "Return rate and lost revenue by product category",
        "examples": ["Which category has the highest return rate?", "How much revenue did returns cost us?"],
        "sql": "SELECT category, COUNT(order_id) AS total_orders, SUM(CASE WHEN returned = 'Yes' THEN 1 ELSE 0 END) AS returned_count, ROUND(100.0 * SUM(CASE WHEN returned = 'Yes' THEN 1 ELSE 0 END) / COUNT(order_id), 2) AS return_rate_pct, SUM(CASE WHEN returned = 'Yes' THEN total_amount ELSE 0 END) AS lost_revenue FROM ecommerce_data GROUP BY category ORDER BY return_rate_pct DESC;"
    },
    {
        "query_id": "q8",
        "description": "Revenue by customer age group and gender",
        "examples": ["Which age group and gender buys the most?", "Revenue by customer demographics"],
        "sql": "SELECT customer_gender, CASE WHEN customer_age < 25 THEN 'Under 25' WHEN customer_age BETWEEN 25 AND 40 THEN '25-40' WHEN customer_age BETWEEN 41 AND 60 THEN '41-60' ELSE 'Above 60' END AS age_group, COUNT(order_id) AS total_orders, SUM(total_amount) AS total_revenue FROM ecommerce_data WHERE returned = 'No' GROUP BY customer_gender, age_group ORDER BY total_revenue DESC;"
    },
    {
        "query_id": "q9",
        "description": "Monthly revenue trend and order count over time",
        "examples": ["How has revenue changed month by month?", "Show the sales trend over time"],
        "sql": "SELECT strftime('%Y-%m', order_date) AS month, COUNT(order_id) AS monthly_orders, SUM(total_amount) AS monthly_revenue FROM ecommerce_data WHERE returned = 'No' GROUP BY month ORDER BY month ASC;"
    },
    {
        "query_id": "q10",
        "description": "High-value orders with discounts above 20 percent",
        "examples": ["Show expensive orders with big discounts", "Which big purchases got heavy discounts?"],
        "sql": "SELECT order_id, customer_id, category, price, discount, total_amount FROM ecommerce_data WHERE discount > 0.20 AND total_amount > 500 ORDER BY total_amount DESC LIMIT 10;"
    },
]

Q_BY_ID = {q["query_id"]: q for q in QUERIES}

# ── Embeddings ────────────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

docs = []
for q in QUERIES:
    docs.append((q["query_id"], q["description"]))
    docs += [(q["query_id"], ex) for ex in q["examples"]]

doc_vectors = embedder.encode([text for _, text in docs], normalize_embeddings=True)
print(f"✅ Embedded {len(docs)} texts")

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a business data analyst.
You receive a user question and the result of a SQL query as a CSV.
Rules:
1. Answer ONLY from the given data — copy numbers exactly, do not round or recompute.
2. Write 2 to 4 clear sentences. Highlight the most important finding first.
3. Do not mention SQL, tables, or technical terms.
4. If the data cannot answer the question, say so honestly."""

# ── Helper functions ──────────────────────────────────────────────────────────

def find_best_query(user_question, top_k=3):
    q_vec = embedder.encode([user_question], normalize_embeddings=True)[0]
    sims  = doc_vectors @ q_vec
    best  = {}
    for (qid, _), s in zip(docs, sims):
        best[qid] = max(best.get(qid, -1.0), float(s))
    ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
    return [{"query_id": qid, "score": round(s, 3), "description": Q_BY_ID[qid]["description"]}
            for qid, s in ranked]


def run_sql(sql):
    if not re.match(r"^\s*select\b", sql, re.IGNORECASE):
        raise ValueError("Only SELECT statements are allowed.")
    uri  = Path(DB_PATH).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def ask_llm(user_question, query_description, result_df):
    data_csv = result_df.round(2).head(30).to_csv(index=False)
    user_message = (
        f"User question: {user_question}\n"
        f"What the query computes: {query_description}\n"
        f"Number of rows returned: {len(result_df)}\n"
        f"\nSQL RESULT (CSV):\n{data_csv}"
    )
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()



def answer_question(question):
    start   = time.time()
    matches = find_best_query(question, top_k=3)
    best    = matches[0]

    if best["score"] < MIN_SIMILARITY:
        topics = "\n".join(f"  • {q['description']}" for q in QUERIES)
        return {"answer": f"❓ Couldn't match your question (score {best['score']:.2f} < {MIN_SIMILARITY}).\n\nI can answer:\n{topics}",
                "df": pd.DataFrame(), "fig": None, "matched": None}

    matched   = Q_BY_ID[best["query_id"]]
    result_df = run_sql(matched["sql"])

    try:
        llm_summary = ask_llm(question, matched["description"], result_df)
    except Exception as e:
        llm_summary = f"⚠️ LLM unavailable ({e}). Data shown below."

    parts = [llm_summary]
    if len(result_df) == 1:
        def fmt(v):
            if isinstance(v, float): return f"{v:,.2f}"
            if isinstance(v, int):   return f"{v:,}"
            return str(v)
        exact = " · ".join(f"**{c}** = {fmt(result_df[c].iloc[0])}" for c in result_df.columns)
        parts.append(f"\n📌 Exact values: {exact}")

    parts.append(f"\n_Matched: **{best['query_id']}** (score {best['score']:.2f}) · {time.time()-start:.1f}s_")

    return {
        "answer":  "\n\n".join(parts),
        "df":      result_df.round(2),
        "fig":     None,
        "matched": matched
    }

# ── Gradio UI ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    "How much money did we make overall?",
    "Which category has the highest return rate?",
    "Who are our top 5 customers?",
    "How is revenue trending month by month?",
    "Which region has the slowest delivery?",
    "What payment methods do customers prefer?",
]

NO_API_KEY_MSG = (
    "⚠️ **GROQ_API_KEY not set.**\n\n"
    "Go to your Space → **Settings → Variables and secrets** → add `GROQ_API_KEY`.\n"
    "Get a free key at https://console.groq.com/keys"
)


def df_to_markdown(df):
    """Convert DataFrame to a markdown table string."""
    if df is None or df.empty:
        return ""
    return df.to_markdown(index=False)


def render_history(history):
    """Convert list of (user, assistant) tuples to an HTML string."""
    if not history:
        return "<p style='color:#888;font-style:italic;padding:12px'>Ask a question to get started.</p>"
    parts = []
    for user_msg, bot_msg in history:
        # Escape HTML in user message
        safe_user = user_msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        # Bot message may contain markdown-style table — render newlines as <br>
        safe_bot = bot_msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        parts.append(
            f'<div style="margin-bottom:14px">'
            f'<div style="background:#e8f0fe;border-radius:8px;padding:10px 14px;margin-bottom:6px">'
            f'<b>🧑 You:</b> {safe_user}</div>'
            f'<div style="background:#f8f9fa;border-radius:8px;padding:10px 14px">'
            f'<b>🤖 Assistant:</b><br>{safe_bot}</div>'
            f'</div>'
        )
    return "".join(parts)


def chat(user_message, history):
    if not user_message or not user_message.strip():
        return history, render_history(history), ""

    if history is None:
        history = []

    if not GROQ_API_KEY:
        history = history + [(user_message, NO_API_KEY_MSG)]
        return history, render_history(history), ""

    res = answer_question(user_message)

    reply = res["answer"]
    if res["df"] is not None and not res["df"].empty:
        reply += "\n\nResult table:\n" + df_to_markdown(res["df"])

    history = history + [(user_message, reply)]
    return history, render_history(history), ""


def clear_all():
    empty_html = "<p style='color:#888;font-style:italic;padding:12px'>Ask a question to get started.</p>"
    return [], empty_html, ""


with gr.Blocks(title="NL → SQL Insight Chatbot") as demo:

    gr.Markdown("""
    # 🤖 NL → SQL Insight Chatbot
    Ask business questions in plain English. The system finds the best pre-written SQL query,
    runs it, and uses a Groq LLM to explain the result.

    > **How it works:** Your question is matched to one of 10 vetted SQL queries using
    > semantic search (cosine similarity). The LLM never writes SQL — it only reads the
    > result and explains it.
    """)

    if not GROQ_API_KEY:
        gr.Markdown(
            "> ⚠️ **No API key detected.** Go to **Settings → Variables and secrets** "
            "→ add `GROQ_API_KEY`. Get a free key at https://console.groq.com/keys"
        )

    # gr.State holds plain Python list — no complex JSON schema, no gradio_client bug
    history_state = gr.State([])
    chat_display = gr.HTML(
        value="<p style='color:#888;font-style:italic;padding:12px'>Ask a question to get started.</p>",
        label="Conversation"
    )

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="e.g. Which region has the most returned orders?",
            show_label=False, scale=5
        )
        send_btn  = gr.Button("Send 🚀", variant="primary", scale=1)
        clear_btn = gr.Button("Clear 🧹", scale=1)

    gr.Examples(
        examples=[[q] for q in EXAMPLES],
        inputs=user_input,
        label="Example questions — click to try"
    )

    outputs = [history_state, chat_display, user_input]
    send_btn.click(chat,  [user_input, history_state], outputs)
    user_input.submit(chat, [user_input, history_state], outputs)
    clear_btn.click(clear_all, None, outputs)

demo.launch(ssr_mode=False)
