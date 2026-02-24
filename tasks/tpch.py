"""
Six TPC-H analytical tasks for the ceiling study.

Tasks are designed to cover three result-size tiers:
  SMALL  (≤10 rows)   — SQL fast, data path negligible
  MEDIUM (50-200 rows) — noticeable data path overhead
  LARGE  (1K+ rows)   — data path may dominate colocation calculation

Each task has:
  - natural_question: what the user asks in plain English
  - schema_hint: relevant tables and columns (given to LLM in system prompt)
  - gold_sql: correct SQL for falling back when LLM fails
  - expected_result_tier: "small" | "medium" | "large"
  - n_sql_steps: expected number of SQL calls needed (1 = direct, 2-3 = multi-step)
"""

from dataclasses import dataclass


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
    # For numeric tasks: the expected value as a string (e.g. "12345.67").
    # For multi-row: a distinctive value from the first result row.
    validation_key: str | None = None


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


# ── Tasks ─────────────────────────────────────────────────────────────────────

TASKS: list[Task] = [

    # ── SMALL tier ────────────────────────────────────────────────────────────

    # Validation keys are computed from gold SQL on SF=1 data.
    # They're checked as substrings in the agent's final answer (case-insensitive).
    # Run setup/compute_validation_keys.py to refresh these for other SFs.

    Task(
        name="q6_discount_revenue",
        natural_question=(
            "What was the total revenue lost due to discounts on shipments "
            "in 1997 where the discount was between 2% and 4% and quantity "
            "was less than 24? Give a single number."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""
            SELECT round(sum(l_extendedprice * l_discount), 2) AS lost_revenue
            FROM lineitem
            WHERE l_shipdate >= DATE '1997-01-01'
              AND l_shipdate  < DATE '1998-01-01'
              AND l_discount BETWEEN 0.02 AND 0.04
              AND l_quantity < 24
        """.strip(),
        expected_result_tier="small",
        n_sql_steps=1,
    ),

    Task(
        name="q1_lineitem_summary",
        natural_question=(
            "Summarize the lineitem table by return flag and line status for "
            "all items shipped on or before August 19, 1995. Show total quantity, "
            "total base price, total discounted price, average quantity, average "
            "price, average discount, and count. Order by flag and status."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""
            SELECT l_returnflag, l_linestatus,
                   sum(l_quantity)                                       AS sum_qty,
                   round(sum(l_extendedprice), 2)                       AS sum_base_price,
                   round(sum(l_extendedprice * (1 - l_discount)), 2)    AS sum_disc_price,
                   round(avg(l_quantity), 2)                            AS avg_qty,
                   round(avg(l_extendedprice), 2)                       AS avg_price,
                   round(avg(l_discount), 4)                            AS avg_disc,
                   count(*)                                             AS count_order
            FROM lineitem
            WHERE l_shipdate <= DATE '1995-08-19'
            GROUP BY l_returnflag, l_linestatus
            ORDER BY l_returnflag, l_linestatus
        """.strip(),
        expected_result_tier="small",
        n_sql_steps=1,
    ),

    # ── MEDIUM tier ───────────────────────────────────────────────────────────

    Task(
        name="q3_unshipped_revenue",
        natural_question=(
            "Find the top 10 unshipped orders with the highest revenue potential "
            "for HOUSEHOLD customers. An order is unshipped if the order was placed "
            "before March 25, 1995 and the items haven't shipped yet (ship date after "
            "that date). Return the order key, total revenue, order date, and ship "
            "priority, sorted by revenue descending."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""
            SELECT l.l_orderkey,
                   round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue,
                   o.o_orderdate,
                   o.o_shippriority
            FROM customer c
            JOIN orders o   ON c.c_custkey = o.o_custkey
            JOIN lineitem l ON l.l_orderkey = o.o_orderkey
            WHERE c.c_mktsegment = 'HOUSEHOLD'
              AND o.o_orderdate  < DATE '1995-03-25'
              AND l.l_shipdate   > DATE '1995-03-25'
            GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority
            ORDER BY revenue DESC, o.o_orderdate
            LIMIT 10
        """.strip(),
        expected_result_tier="small",
        n_sql_steps=1,
    ),

    Task(
        name="q5_europe_revenue",
        natural_question=(
            "What is the total revenue generated by each nation in the EUROPE region "
            "from local suppliers during 1997? (A sale is 'local' when the customer "
            "and the supplier are in the same nation.) "
            "Return nation name and revenue, sorted by revenue descending."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""
            SELECT n.n_name,
                   round(sum(l.l_extendedprice * (1 - l.l_discount)), 2) AS revenue
            FROM orders o
            JOIN lineitem l  ON l.l_orderkey = o.o_orderkey
            JOIN supplier s  ON l.l_suppkey  = s.s_suppkey
            JOIN nation n    ON s.s_nationkey = n.n_nationkey
            JOIN region r    ON n.n_regionkey = r.r_regionkey
            JOIN customer c  ON o.o_custkey   = c.c_custkey
                             AND c.c_nationkey = s.s_nationkey
            WHERE r.r_name = 'EUROPE'
              AND o.o_orderdate >= DATE '1997-01-01'
              AND o.o_orderdate  < DATE '1998-01-01'
            GROUP BY n.n_name
            ORDER BY revenue DESC
        """.strip(),
        expected_result_tier="small",
        n_sql_steps=2,
    ),

    # ── LARGE tier ────────────────────────────────────────────────────────────

    Task(
        name="q16_supplier_count",
        natural_question=(
            "For parts that are NOT Brand#21 and NOT of a type starting with "
            "'MEDIUM PLATED', and whose size is in (38,2,8,31,44,5,14,24): "
            "count how many distinct suppliers can provide each brand/type/size "
            "combination, excluding suppliers who have customer complaints on record. "
            "Return the top 20 combinations by supplier count."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""
            SELECT p.p_brand, p.p_type, p.p_size,
                   count(DISTINCT ps.ps_suppkey) AS supplier_cnt
            FROM partsupp ps
            JOIN part p ON p.p_partkey = ps.ps_partkey
            WHERE p.p_brand <> 'Brand#21'
              AND p.p_type  NOT LIKE 'MEDIUM PLATED%'
              AND p.p_size  IN (38, 2, 8, 31, 44, 5, 14, 24)
              AND ps.ps_suppkey NOT IN (
                  SELECT s.s_suppkey FROM supplier s
                  WHERE s.s_comment LIKE '%Customer%Complaints%'
              )
            GROUP BY p.p_brand, p.p_type, p.p_size
            ORDER BY supplier_cnt DESC, p.p_brand, p.p_type, p.p_size
            LIMIT 20
        """.strip(),
        expected_result_tier="medium",
        n_sql_steps=2,
    ),

    Task(
        name="large_shipment_scan",
        natural_question=(
            "Retrieve the 1000 most recent shipments from the lineitem table, "
            "showing order key, part key, supplier key, quantity, extended price, "
            "discount, return flag, line status, and ship date. "
            "I need to do some downstream analysis of these records."
        ),
        schema_hint=TPCH_SCHEMA,
        gold_sql="""
            SELECT l_orderkey, l_partkey, l_suppkey,
                   l_quantity, l_extendedprice, l_discount,
                   l_returnflag, l_linestatus, l_shipdate
            FROM lineitem
            ORDER BY l_shipdate DESC
            LIMIT 1000
        """.strip(),
        expected_result_tier="large",
        n_sql_steps=1,
    ),
]

TASKS_BY_NAME: dict[str, Task] = {t.name: t for t in TASKS}


def get_task(name: str) -> Task:
    if name not in TASKS_BY_NAME:
        raise KeyError(f"Unknown task: {name!r}. Available: {list(TASKS_BY_NAME)}")
    return TASKS_BY_NAME[name]
