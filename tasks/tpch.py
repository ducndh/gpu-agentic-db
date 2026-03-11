"""
TPC-H analytical tasks for the GPU colocation ceiling study.

Task design goals:
  - Cover realistic agentic patterns: schema discovery, multi-step analysis,
    error recovery, exploratory analysis, complex SQL (CTEs, window functions)
  - Cover three result-size tiers to show how data-path scales:
      SMALL  (≤10 rows)   — minimal serialization overhead
      MEDIUM (50-200 rows) — noticeable data-path cost
      LARGE  (1K+ rows)   — data-path dominates colocation ceiling
  - Multi-step tasks force the agent to issue multiple SQL queries per attempt,
    which multiplies the data-path overhead relative to LLM cost

Each task has:
  - natural_question: what the user asks in plain English
  - schema_hint: relevant tables and columns (given to LLM in system prompt)
  - gold_sql: correct SQL for validation and fallback
  - expected_result_tier: "small" | "medium" | "large"
  - n_sql_steps: expected number of SQL calls the agent should make
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    name: str
    natural_question: str
    schema_hint: str
    gold_sql: str
    expected_result_tier: str   # "small" | "medium" | "large"
    n_sql_steps: int             # expected number of SQL calls
    # validation_key: a string that MUST appear in the agent's final answer
    # to count as correct (case-insensitive). None = no validation.
    validation_key: Optional[str] = None


# ── Shared schema description ─────────────────────────────────────────────────

TPCH_SCHEMA = """
TPC-H database schema (read-only, all tables already loaded):

  lineitem(l_orderkey, l_partkey, l_suppkey, l_linenumber, l_quantity,
           l_extendedprice, l_discount, l_tax, l_returnflag, l_linestatus,
           l_shipdate, l_commitdate, l_receiptdate, l_shipinstruct,
           l_shipmode, l_comment)

  orders(o_orderkey, o_custkey, o_orderstatus, o_totalprice, o_orderdate,
         o_orderpriority, o_clerk, o_shippriority, o_comment)

  customer(c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal,
           c_mktsegment, c_comment)

  supplier(s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal,
           s_comment)

  part(p_partkey, p_name, p_mfgr, p_brand, p_type, p_size, p_container,
       p_retailprice, p_comment)

  partsupp(ps_partkey, ps_suppkey, ps_availqty, ps_supplycost, ps_comment)

  nation(n_nationkey, n_name, n_regionkey, n_comment)
  region(r_regionkey, r_name, r_comment)

Dates are stored as DATE type. Revenue = l_extendedprice * (1 - l_discount).
""".strip()

# Minimal schema hint for discovery tasks — forces the agent to explore
MINIMAL_SCHEMA = (
    "You have access to a TPC-H database. "
    "Use SHOW TABLES and DESCRIBE <table> to explore the schema before querying."
)


# ── Tasks ─────────────────────────────────────────────────────────────────────

TASKS: list[Task] = [

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 1: Schema Discovery — agent must explore before querying
    # These tasks give minimal schema hints, forcing SHOW TABLES / DESCRIBE
    # calls that add 1-3 extra SQL turns (more data-path overhead).
    # ══════════════════════════════════════════════════════════════════════════

    Task(
        name="discovery_revenue_1997",
        natural_question=(
            "I have a database but I'm not sure what's in it. "
            "Can you explore the schema and then tell me the total revenue "
            "from orders in 1997? Revenue is extended price times (1 - discount). "
            "Give me a single number."
        ),
        schema_hint=MINIMAL_SCHEMA,
        gold_sql="""\
SELECT round(sum(l_extendedprice * (1 - l_discount)), 2) AS revenue
FROM lineitem l
JOIN orders o ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= DATE '1997-01-01'
  AND o.o_orderdate < DATE '1998-01-01'""",
        expected_result_tier="small",
        n_sql_steps=3,  # SHOW TABLES → DESCRIBE lineitem/orders → query
    ),

    Task(
        name="discovery_top_nations",
        natural_question=(
            "I don't know this database. Explore the tables and tell me "
            "which 5 nations have the most suppliers. Show nation name and count."
        ),
        schema_hint=MINIMAL_SCHEMA,
        gold_sql="""\
SELECT n.n_name, count(*) AS supplier_count
FROM supplier s
JOIN nation n ON s.s_nationkey = n.n_nationkey
GROUP BY n.n_name
ORDER BY supplier_count DESC
LIMIT 5""",
        expected_result_tier="small",
        n_sql_steps=3,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 2: Multi-Step Dependent Queries — step 2 depends on step 1
    # The agent must chain queries: find something, then drill into it.
    # Each step adds a full data-path cycle.
    # ══════════════════════════════════════════════════════════════════════════

    Task(
        name="multistep_top_customer_orders",
        natural_question=(
            "Find the customer who has spent the most money overall (by total order value). "
            "Then show me all of that customer's orders with their dates and total prices, "
            "sorted by date. I want to see the full order history."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
SELECT o_orderkey, o_orderdate, o_totalprice
FROM orders
WHERE o_custkey = (
    SELECT o_custkey FROM orders
    GROUP BY o_custkey
    ORDER BY sum(o_totalprice) DESC
    LIMIT 1
)
ORDER BY o_orderdate""",
        expected_result_tier="medium",
        n_sql_steps=2,  # find top customer → get their orders
    ),

    Task(
        name="multistep_supplier_analysis",
        natural_question=(
            "Which supplier in EUROPE has the highest total revenue from lineitem? "
            "Once you find them, show me a breakdown of their revenue by year. "
            "Revenue = l_extendedprice * (1 - l_discount)."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
WITH top_supplier AS (
    SELECT l.l_suppkey,
           round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS total_rev
    FROM lineitem l
    JOIN supplier s ON l.l_suppkey = s.s_suppkey
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    JOIN region r ON n.n_regionkey = r.r_regionkey
    WHERE r.r_name = 'EUROPE'
    GROUP BY l.l_suppkey
    ORDER BY total_rev DESC
    LIMIT 1
)
SELECT extract(year FROM l.l_shipdate) AS year,
       round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue
FROM lineitem l
WHERE l.l_suppkey = (SELECT l_suppkey FROM top_supplier)
GROUP BY year
ORDER BY year""",
        expected_result_tier="small",
        n_sql_steps=2,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 3: Complex Analytical Queries — CTEs, window functions, subqueries
    # Tests whether the agent can write sophisticated SQL in one shot.
    # Large result sets amplify the data-path overhead.
    # ══════════════════════════════════════════════════════════════════════════

    Task(
        name="ranked_suppliers_by_nation",
        natural_question=(
            "Rank all suppliers by their total revenue within each nation. "
            "Show the top 3 suppliers per nation with their rank, name, nation, "
            "and total revenue. Revenue = l_extendedprice * (1 - l_discount)."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
WITH supplier_rev AS (
    SELECT s.s_suppkey, s.s_name, n.n_name AS nation,
           round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue
    FROM lineitem l
    JOIN supplier s ON l.l_suppkey = s.s_suppkey
    JOIN nation n ON s.s_nationkey = n.n_nationkey
    GROUP BY s.s_suppkey, s.s_name, n.n_name
)
SELECT * FROM (
    SELECT *, row_number() OVER (PARTITION BY nation ORDER BY revenue DESC) AS rnk
    FROM supplier_rev
) WHERE rnk <= 3
ORDER BY nation, rnk""",
        expected_result_tier="medium",  # 25 nations × 3 = 75 rows
        n_sql_steps=1,
    ),

    Task(
        name="monthly_trend_analysis",
        natural_question=(
            "Show me the monthly revenue trend for all of 1996 and 1997 side by side. "
            "For each month, I want: month, year-1996 revenue, year-1997 revenue, "
            "and the percent change. Revenue = l_extendedprice * (1 - l_discount)."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
WITH monthly AS (
    SELECT extract(month FROM l.l_shipdate) AS month,
           extract(year FROM l.l_shipdate) AS year,
           round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue
    FROM lineitem l
    WHERE l.l_shipdate >= DATE '1996-01-01'
      AND l.l_shipdate < DATE '1998-01-01'
    GROUP BY month, year
)
SELECT m96.month,
       m96.revenue AS rev_1996,
       m97.revenue AS rev_1997,
       round((m97.revenue - m96.revenue) / m96.revenue * 100, 1) AS pct_change
FROM monthly m96
JOIN monthly m97 ON m96.month = m97.month
WHERE m96.year = 1996 AND m97.year = 1997
ORDER BY m96.month""",
        expected_result_tier="small",  # 12 rows
        n_sql_steps=1,
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 4: Exploratory / Vague Questions — agent decides what to query
    # No single right answer; agent must run multiple exploratory queries.
    # More turns = more data-path overhead.
    # ══════════════════════════════════════════════════════════════════════════

    Task(
        name="exploratory_segment_analysis",
        natural_question=(
            "Analyze the performance of different customer market segments. "
            "I want to understand which segments generate the most revenue, "
            "have the most orders, and have the highest average order value. "
            "Also show how many customers are in each segment."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
SELECT c.c_mktsegment AS segment,
       count(DISTINCT c.c_custkey) AS n_customers,
       count(DISTINCT o.o_orderkey) AS n_orders,
       round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS total_revenue,
       round(avg(o.o_totalprice), 2) AS avg_order_value,
       round(sum(l.l_extendedprice * (1 - l.l_discount)) / count(DISTINCT c.c_custkey), 2) AS revenue_per_customer
FROM customer c
JOIN orders o ON c.c_custkey = o.o_custkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
GROUP BY c.c_mktsegment
ORDER BY total_revenue DESC""",
        expected_result_tier="small",  # 5 segments
        n_sql_steps=2,  # might explore, then aggregate
    ),

    # ══════════════════════════════════════════════════════════════════════════
    # PATTERN 5: Large Result Sets — maximize data-path overhead
    # These queries return hundreds or thousands of rows, which makes
    # fetch + serialize + tokenize a larger fraction of total time.
    # ══════════════════════════════════════════════════════════════════════════

    Task(
        name="large_shipment_scan",
        natural_question=(
            "Retrieve the 1000 most recent shipments from the lineitem table, "
            "showing order key, part key, supplier key, quantity, extended price, "
            "discount, return flag, line status, and ship date. "
            "I need to do some downstream analysis of these records."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
SELECT l_orderkey, l_partkey, l_suppkey,
       l_quantity, l_extendedprice, l_discount,
       l_returnflag, l_linestatus, l_shipdate
FROM lineitem
ORDER BY l_shipdate DESC
LIMIT 1000""",
        expected_result_tier="large",
        n_sql_steps=1,
    ),

    Task(
        name="large_supplier_parts",
        natural_question=(
            "Show me a comprehensive list of all part-supplier combinations "
            "where the supply cost is above the average supply cost for that part. "
            "Include supplier name, part name, supply cost, available quantity, "
            "and the nation of the supplier. Limit to 500 rows, ordered by "
            "supply cost descending."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
SELECT s.s_name, p.p_name, ps.ps_supplycost, ps.ps_availqty, n.n_name
FROM partsupp ps
JOIN supplier s ON ps.ps_suppkey = s.s_suppkey
JOIN part p ON ps.ps_partkey = p.p_partkey
JOIN nation n ON s.s_nationkey = n.n_nationkey
WHERE ps.ps_supplycost > (
    SELECT avg(ps2.ps_supplycost)
    FROM partsupp ps2
    WHERE ps2.ps_partkey = ps.ps_partkey
)
ORDER BY ps.ps_supplycost DESC
LIMIT 500""",
        expected_result_tier="large",
        n_sql_steps=1,
    ),

    Task(
        name="large_order_details",
        natural_question=(
            "Pull a detailed report of all orders from BUILDING segment customers "
            "in 1997. For each order, show: order key, customer name, nation, "
            "order date, total price, number of line items, and total revenue. "
            "Sort by total revenue descending, limit to 500."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""\
SELECT o.o_orderkey, c.c_name, n.n_name AS nation,
       o.o_orderdate, o.o_totalprice,
       count(l.l_linenumber) AS n_items,
       round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue
FROM orders o
JOIN customer c ON o.o_custkey = c.c_custkey
JOIN nation n ON c.c_nationkey = n.n_nationkey
JOIN lineitem l ON o.o_orderkey = l.l_orderkey
WHERE c.c_mktsegment = 'BUILDING'
  AND o.o_orderdate >= DATE '1997-01-01'
  AND o.o_orderdate < DATE '1998-01-01'
GROUP BY o.o_orderkey, c.c_name, n.n_name, o.o_orderdate, o.o_totalprice
ORDER BY revenue DESC
LIMIT 500""",
        expected_result_tier="large",
        n_sql_steps=1,
    ),

]

TASKS_BY_NAME: dict[str, Task] = {t.name: t for t in TASKS}

# Map tasks to their result size tier for analysis grouping
TIER_MAP: dict[str, str] = {t.name: t.expected_result_tier for t in TASKS}


def get_task(name: str) -> Task:
    if name not in TASKS_BY_NAME:
        raise KeyError(f"Unknown task: {name!r}. Available: {list(TASKS_BY_NAME)}")
    return TASKS_BY_NAME[name]
