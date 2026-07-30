-- ============================================================================
-- business_analytics.sql
--
-- Phase 8 of the pipeline: a curated library of business-oriented SQL
-- queries against the normalized schema from sql/schema.sql. Where
-- feature_engineering.py exists so Python-side analysis (statistics.py,
-- forecasting.py, the dashboard) never has to re-join the eight tables by
-- hand, this file is the equivalent reference for doing that analysis
-- directly in SQL -- the layer an analytics-company interview is most
-- likely to probe directly, independent of the Python code.
--
-- This file is a reference library, not a pipeline step -- nothing in
-- src/ executes it automatically. Run individual queries with:
--   psql -U postgres -d retail_analytics -f sql/business_analytics.sql
-- or copy a single query into any SQL client / notebook.
--
-- Every query follows the same three-part structure: the business
-- question it answers, the query itself, and what pattern in the result
-- would actually be worth acting on. That structure matters more than the
-- SQL syntax -- knowing why a query exists is what turns "I can write a
-- window function" into "I can tell you what to do with the answer."
--
-- Organized in the order a real analytics engagement would move through
-- them: start with basic aggregation, add conditional logic, then layer on
-- CTEs and window functions for the analysis that basic GROUP BY can't
-- express (rankings, running totals, period-over-period growth), then
-- close with three purpose-built analyses (customer, product/region,
-- RFM) that combine everything above.
-- ============================================================================


-- ============================================================================
-- SECTION 1: BASIC AGGREGATIONS
-- ============================================================================

-- Query 1: Headline business KPIs
-- Objective: The single query a business review meeting would open with --
--            total revenue, profit, orders, and derived margin/AOV in one row.
-- Expected insight: A baseline to compare every other query's numbers against,
--            and to track period over period.
SELECT
    ROUND(SUM(s.sales)::numeric, 2)                          AS total_revenue,
    ROUND(SUM(s.profit)::numeric, 2)                         AS total_profit,
    ROUND((SUM(s.profit) / NULLIF(SUM(s.sales), 0) * 100)::numeric, 2) AS profit_margin_pct,
    COUNT(DISTINCT s.order_id)                                AS total_orders,
    ROUND((SUM(s.sales) / NULLIF(COUNT(DISTINCT s.order_id), 0))::numeric, 2) AS avg_order_value
FROM sales s;


-- Query 2: Revenue and profit by category
-- Objective: Which product categories drive the business, in absolute terms.
-- Expected insight: Categories are rarely proportional -- the category with
--            the most revenue is often not the one with the best margin,
--            which is the first thing worth flagging to a category manager.
SELECT
    c.category_name,
    COUNT(*)                                    AS line_items,
    ROUND(SUM(s.sales)::numeric, 2)             AS revenue,
    ROUND(SUM(s.profit)::numeric, 2)            AS profit,
    ROUND((SUM(s.profit) / NULLIF(SUM(s.sales), 0) * 100)::numeric, 2) AS margin_pct
FROM sales s
JOIN products p       ON s.product_id = p.product_id
JOIN sub_categories sc ON p.subcategory_id = sc.subcategory_id
JOIN categories c      ON sc.category_id = c.category_id
GROUP BY c.category_name
ORDER BY revenue DESC;


-- Query 3: Revenue and profit by region
-- Objective: Geographic performance -- where the business is concentrated.
-- Expected insight: A region contributing a large revenue share but a
--            below-average margin share is worth a pricing or cost review
--            (see also Section 11, profitability analysis).
SELECT
    r.region_name,
    COUNT(*)                          AS line_items,
    ROUND(SUM(s.sales)::numeric, 2)   AS revenue,
    ROUND(SUM(s.profit)::numeric, 2)  AS profit,
    ROUND((SUM(s.profit) / NULLIF(SUM(s.sales), 0) * 100)::numeric, 2) AS margin_pct
FROM sales s
JOIN orders o    ON s.order_id = o.order_id
JOIN locations l ON o.location_id = l.location_id
JOIN regions r   ON l.region_id = r.region_id
GROUP BY r.region_name
ORDER BY revenue DESC;


-- Query 4: Overall discount and margin profile
-- Objective: A single-row sanity check on how aggressively the business
--            discounts, and what that costs in margin on average.
-- Expected insight: A baseline average discount/margin pair to compare
--            every category- or region-level discount pattern against.
SELECT
    ROUND(AVG(s.discount)::numeric, 4)       AS avg_discount,
    ROUND(AVG(s.profit / NULLIF(s.sales,0))::numeric, 4) AS avg_margin,
    ROUND(MIN(s.discount)::numeric, 2)       AS min_discount,
    ROUND(MAX(s.discount)::numeric, 2)       AS max_discount
FROM sales s;


-- ============================================================================
-- SECTION 2: GROUP BY + HAVING
-- ============================================================================

-- Query 5: Sub-categories with thin average margin
-- Objective: Surface sub-categories whose average margin is below 8% --
--            candidates for a pricing review, filtered with HAVING rather
--            than a WHERE clause since the filter is on an aggregate.
-- Expected insight: A short, actionable list rather than all 17
--            sub-categories -- HAVING does the "worth looking at" filtering
--            that a WHERE clause on raw rows cannot express.
SELECT
    sc.subcategory_name,
    ROUND(SUM(s.sales)::numeric, 2)  AS revenue,
    ROUND(AVG(s.profit / NULLIF(s.sales,0)) * 100, 2) AS avg_margin_pct
FROM sales s
JOIN products p        ON s.product_id = p.product_id
JOIN sub_categories sc ON p.subcategory_id = sc.subcategory_id
GROUP BY sc.subcategory_name
HAVING AVG(s.profit / NULLIF(s.sales,0)) < 0.08
ORDER BY avg_margin_pct ASC;


-- Query 6: High-frequency customers
-- Objective: Customers who have placed more than 3 distinct orders --
--            the business's most engaged repeat buyers.
-- Expected insight: A shortlist for a loyalty program or account-management
--            outreach, rather than treating all customers identically.
SELECT
    c.customer_id,
    c.customer_name,
    c.segment,
    COUNT(DISTINCT o.order_id) AS order_count,
    ROUND(SUM(s.sales)::numeric, 2) AS lifetime_revenue
FROM sales s
JOIN orders o    ON s.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.customer_id, c.customer_name, c.segment
HAVING COUNT(DISTINCT o.order_id) > 3
ORDER BY order_count DESC;


-- Query 7: Products with broad order reach
-- Objective: Products purchased in at least 5 distinct orders -- i.e.
--            reliably repeat-purchased rather than a one-off spike.
-- Expected insight: These are the safest products to keep well-stocked --
--            demand for them isn't dependent on one large one-time order.
SELECT
    p.product_name,
    COUNT(DISTINCT s.order_id) AS distinct_orders,
    SUM(s.quantity)            AS total_units_sold
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
HAVING COUNT(DISTINCT s.order_id) >= 5
ORDER BY distinct_orders DESC;


-- Query 8: Regions operating at a net loss
-- Objective: Any region where total profit across all sales is negative.
-- Expected insight: With HAVING SUM(profit) < 0, an empty result is itself
--            a meaningful (positive) finding -- it means no region as a
--            whole is loss-making, even if individual products within it are.
SELECT
    r.region_name,
    ROUND(SUM(s.sales)::numeric, 2)  AS revenue,
    ROUND(SUM(s.profit)::numeric, 2) AS profit
FROM sales s
JOIN orders o    ON s.order_id = o.order_id
JOIN locations l ON o.location_id = l.location_id
JOIN regions r   ON l.region_id = r.region_id
GROUP BY r.region_name
HAVING SUM(s.profit) < 0
ORDER BY profit ASC;


-- ============================================================================
-- SECTION 3: CASE WHEN
-- ============================================================================

-- Query 9: Discount tier analysis
-- Objective: Bucket every sale into a discount tier, then compare margin
--            and volume across tiers in one query.
-- Expected insight: The tier where avg_margin_pct crosses from positive to
--            negative is a concrete, defensible "don't discount past this"
--            number -- the same logic recommendations.recommend_discount_
--            threshold() computes in Python, expressed natively in SQL.
SELECT
    CASE
        WHEN s.discount = 0        THEN 'No discount'
        WHEN s.discount <= 0.10    THEN 'Low (0-10%)'
        WHEN s.discount <= 0.20    THEN 'Medium (11-20%)'
        ELSE                            'High (21%+)'
    END AS discount_tier,
    COUNT(*)                         AS line_items,
    ROUND(SUM(s.sales)::numeric, 2)  AS revenue,
    ROUND(AVG(s.profit / NULLIF(s.sales,0)) * 100, 2) AS avg_margin_pct
FROM sales s
GROUP BY discount_tier
ORDER BY MIN(s.discount);


-- Query 10: Shipping speed classification
-- Objective: Classify each order by how many days elapsed between order and
--            ship date, then see how volume is distributed across speeds.
-- Expected insight: If "Same day" and "1-3 days" orders are rare relative to
--            "4+ days," fulfillment speed may be a bigger differentiator to
--            offer customers than the business currently markets.
SELECT
    CASE
        WHEN (o.ship_date - o.order_date) = 0                    THEN 'Same day'
        WHEN (o.ship_date - o.order_date) BETWEEN 1 AND 3         THEN '1-3 days'
        ELSE                                                           '4+ days'
    END AS shipping_speed,
    COUNT(DISTINCT o.order_id) AS order_count,
    ROUND(AVG(o.ship_date - o.order_date), 2) AS avg_days_to_ship
FROM orders o
GROUP BY shipping_speed
ORDER BY avg_days_to_ship;


-- Query 11: Profitable vs. loss-making sales
-- Objective: Split every line item into profitable vs. loss-making, and
--            quantify how much revenue each group represents.
-- Expected insight: If loss-making sales are a small share of line items
--            but a large share of revenue, a handful of high-value deals
--            are disproportionately eroding overall profit.
SELECT
    CASE WHEN s.profit >= 0 THEN 'Profitable' ELSE 'Loss-making' END AS profitability,
    COUNT(*)                          AS line_items,
    ROUND(SUM(s.sales)::numeric, 2)   AS revenue,
    ROUND(SUM(s.profit)::numeric, 2)  AS profit,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_line_items
FROM sales s
GROUP BY profitability;


-- ============================================================================
-- SECTION 4: CTEs
-- ============================================================================

-- Query 12: Each category's monthly share of total revenue
-- Objective: Using a CTE to first compute monthly totals per category, then
--            a second pass to express each category's share of that month.
-- Expected insight: A category with a shrinking month-over-month share
--            (even while its absolute revenue holds steady) is losing
--            ground relative to the rest of the business -- a signal
--            a flat trend line alone wouldn't show.
WITH monthly_category_revenue AS (
    SELECT
        DATE_TRUNC('month', o.order_date) AS order_month,
        c.category_name,
        SUM(s.sales) AS revenue
    FROM sales s
    JOIN orders o           ON s.order_id = o.order_id
    JOIN products p         ON s.product_id = p.product_id
    JOIN sub_categories sc  ON p.subcategory_id = sc.subcategory_id
    JOIN categories c       ON sc.category_id = c.category_id
    GROUP BY order_month, c.category_name
)
SELECT
    order_month,
    category_name,
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(100.0 * revenue / SUM(revenue) OVER (PARTITION BY order_month), 2) AS pct_of_month_revenue
FROM monthly_category_revenue
ORDER BY order_month, revenue DESC;


-- Query 13: Top 5 products by profit, with their category and region context
-- Objective: A CTE isolates the top-5-by-profit product list; the outer
--            query enriches it with dimension context the CTE didn't need
--            to carry, keeping the ranking logic and the enrichment logic
--            visually separate.
-- Expected insight: If the top-5 list clusters into one or two categories,
--            that's where merchandising/marketing investment is already
--            paying off best.
WITH top_products AS (
    SELECT
        p.product_id,
        p.product_name,
        SUM(s.profit) AS total_profit
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_id, p.product_name
    ORDER BY total_profit DESC
    LIMIT 5
)
SELECT
    tp.product_name,
    ROUND(tp.total_profit::numeric, 2) AS total_profit,
    c.category_name,
    sc.subcategory_name
FROM top_products tp
JOIN products p        ON tp.product_id = p.product_id
JOIN sub_categories sc ON p.subcategory_id = sc.subcategory_id
JOIN categories c      ON sc.category_id = c.category_id
ORDER BY tp.total_profit DESC;


-- Query 14: Top-decile customers by lifetime spend
-- Objective: A CTE computes lifetime spend per customer; the outer query
--            filters to the top 10% using a subquery-based percentile
--            cutoff, avoiding a hardcoded row count that would silently
--            go stale as the customer base grows.
-- Expected insight: This is the customer list a VIP or high-touch account
--            program should be built around -- and, compared against
--            total revenue, shows how concentrated revenue is in a small
--            share of the customer base.
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        SUM(s.sales) AS lifetime_spend
    FROM sales s
    JOIN orders o    ON s.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_name, c.segment
)
SELECT *
FROM customer_spend
WHERE lifetime_spend >= (
    SELECT PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY lifetime_spend)
    FROM customer_spend
)
ORDER BY lifetime_spend DESC;


-- ============================================================================
-- SECTION 5: WINDOW FUNCTIONS & RANKING
-- ============================================================================

-- Query 15: Rank customers by spend within their own segment
-- Objective: RANK() customers by total spend, partitioned by segment, so
--            "top customer" is relative to peers in the same segment
--            rather than the whole customer base.
-- Expected insight: The #1-ranked Home Office customer may be worth far
--            less in absolute dollars than a mid-ranked Corporate customer
--            -- this avoids comparing customers across segments unfairly.
SELECT
    c.segment,
    c.customer_name,
    ROUND(SUM(s.sales)::numeric, 2) AS total_spend,
    RANK() OVER (PARTITION BY c.segment ORDER BY SUM(s.sales) DESC) AS spend_rank_in_segment
FROM sales s
JOIN orders o    ON s.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.segment, c.customer_name
ORDER BY c.segment, spend_rank_in_segment;


-- Query 16: Each customer's most recent order
-- Objective: ROW_NUMBER() partitioned by customer, ordered by order date
--            descending, isolates exactly one row (the latest order) per
--            customer -- the standard "latest record per group" pattern.
-- Expected insight: Feeds directly into recency-based outreach (see also
--            the Recency component of the RFM analysis in Section 12).
WITH ranked_orders AS (
    SELECT
        o.customer_id,
        o.order_id,
        o.order_date,
        ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date DESC) AS rn
    FROM orders o
)
SELECT c.customer_name, ro.order_id, ro.order_date
FROM ranked_orders ro
JOIN customers c ON ro.customer_id = c.customer_id
WHERE ro.rn = 1
ORDER BY ro.order_date DESC;


-- Query 17: Best-selling product per category
-- Objective: DENSE_RANK() products by profit within each category, then
--            keep only the #1 product per category.
-- Expected insight: One "hero product" per category -- useful for deciding
--            what to feature first in category-level marketing.
WITH ranked_products AS (
    SELECT
        c.category_name,
        p.product_name,
        SUM(s.profit) AS total_profit,
        DENSE_RANK() OVER (PARTITION BY c.category_name ORDER BY SUM(s.profit) DESC) AS profit_rank
    FROM sales s
    JOIN products p        ON s.product_id = p.product_id
    JOIN sub_categories sc ON p.subcategory_id = sc.subcategory_id
    JOIN categories c      ON sc.category_id = c.category_id
    GROUP BY c.category_name, p.product_name
)
SELECT category_name, product_name, ROUND(total_profit::numeric, 2) AS total_profit
FROM ranked_products
WHERE profit_rank = 1
ORDER BY total_profit DESC;


-- Query 18: Customer spend quartiles
-- Objective: NTILE(4) splits customers into four equal-sized groups by
--            lifetime spend -- quick, even segmentation without picking
--            arbitrary dollar cutoffs by hand.
-- Expected insight: Quartile 4 (the split's bottom 25% by spend) is a
--            reactivation-campaign candidate list; quartile 1 (top 25%)
--            is the VIP list -- both defined consistently by the same rule.
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.customer_name,
        SUM(s.sales) AS lifetime_spend
    FROM sales s
    JOIN orders o    ON s.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_name
)
SELECT
    customer_name,
    ROUND(lifetime_spend::numeric, 2) AS lifetime_spend,
    NTILE(4) OVER (ORDER BY lifetime_spend DESC) AS spend_quartile
FROM customer_spend
ORDER BY lifetime_spend DESC;


-- Query 19: Change in order value vs. each customer's previous order
-- Objective: LAG() looks back one row (per customer, ordered by date) to
--            compare each order's revenue against that same customer's
--            immediately preceding order.
-- Expected insight: A customer whose order values are trending down over
--            successive orders is a churn-risk signal well before they
--            stop ordering entirely.
WITH customer_orders AS (
    SELECT
        o.customer_id,
        o.order_id,
        o.order_date,
        SUM(s.sales) AS order_value
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    GROUP BY o.customer_id, o.order_id, o.order_date
)
SELECT
    c.customer_name,
    co.order_date,
    ROUND(co.order_value::numeric, 2) AS order_value,
    ROUND(LAG(co.order_value) OVER (PARTITION BY co.customer_id ORDER BY co.order_date)::numeric, 2) AS previous_order_value,
    ROUND((co.order_value - LAG(co.order_value) OVER (PARTITION BY co.customer_id ORDER BY co.order_date))::numeric, 2) AS change_vs_previous
FROM customer_orders co
JOIN customers c ON co.customer_id = c.customer_id
ORDER BY c.customer_name, co.order_date;


-- ============================================================================
-- SECTION 6: RUNNING TOTALS & ROLLING AVERAGES
-- ============================================================================

-- Query 20: Cumulative revenue over time
-- Objective: A running total of monthly revenue using SUM() OVER an
--            ordered, unbounded-preceding window -- the standard
--            "revenue to date" cumulative chart.
-- Expected insight: The slope of the cumulative line is easier to compare
--            visually across different periods than month-to-month bars,
--            since it smooths out single-month noise.
WITH monthly_revenue AS (
    SELECT DATE_TRUNC('month', o.order_date) AS order_month, SUM(s.sales) AS revenue
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    GROUP BY order_month
)
SELECT
    order_month,
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(SUM(revenue) OVER (ORDER BY order_month)::numeric, 2) AS cumulative_revenue
FROM monthly_revenue
ORDER BY order_month;


-- Query 21: 3-month rolling average of revenue
-- Objective: A moving average using ROWS BETWEEN 2 PRECEDING AND CURRENT
--            ROW -- the same smoothing idea behind
--            forecasting.moving_average_forecast(), expressed as a window
--            function instead of a Python loop.
-- Expected insight: Where the rolling average and the raw monthly figure
--            diverge sharply, that month was unusually strong or weak
--            relative to its recent trend -- worth a footnote in reporting.
WITH monthly_revenue AS (
    SELECT DATE_TRUNC('month', o.order_date) AS order_month, SUM(s.sales) AS revenue
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    GROUP BY order_month
)
SELECT
    order_month,
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(AVG(revenue) OVER (ORDER BY order_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2) AS rolling_3mo_avg
FROM monthly_revenue
ORDER BY order_month;


-- Query 22: Running total of profit, by category
-- Objective: Same running-total pattern as Query 20, but PARTITION BY
--            category so each category accumulates independently within
--            the same result set.
-- Expected insight: Compares how quickly each category's cumulative
--            profit grows -- a category with a steep, steady climb is a
--            more dependable profit source than one with a single spike.
WITH monthly_category_profit AS (
    SELECT
        DATE_TRUNC('month', o.order_date) AS order_month,
        c.category_name,
        SUM(s.profit) AS profit
    FROM sales s
    JOIN orders o           ON s.order_id = o.order_id
    JOIN products p         ON s.product_id = p.product_id
    JOIN sub_categories sc  ON p.subcategory_id = sc.subcategory_id
    JOIN categories c       ON sc.category_id = c.category_id
    GROUP BY order_month, c.category_name
)
SELECT
    order_month,
    category_name,
    ROUND(profit::numeric, 2) AS profit,
    ROUND(SUM(profit) OVER (PARTITION BY category_name ORDER BY order_month)::numeric, 2) AS cumulative_profit
FROM monthly_category_profit
ORDER BY category_name, order_month;


-- ============================================================================
-- SECTION 7: YoY / MoM GROWTH
-- ============================================================================

-- Query 23: Month-over-month revenue growth %
-- Objective: LAG() one row back (one calendar month) to compute percentage
--            change from the previous month.
-- Expected insight: Highlights volatility a cumulative chart (Query 20)
--            smooths away -- useful for catching a single bad month before
--            it's buried in an otherwise-fine year.
WITH monthly_revenue AS (
    SELECT DATE_TRUNC('month', o.order_date) AS order_month, SUM(s.sales) AS revenue
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    GROUP BY order_month
)
SELECT
    order_month,
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY order_month)) / NULLIF(LAG(revenue) OVER (ORDER BY order_month), 0), 2) AS mom_growth_pct
FROM monthly_revenue
ORDER BY order_month;


-- Query 24: Year-over-year revenue growth %
-- Objective: Aggregate to the year grain first, then LAG() one row back
--            (one full year) for a growth rate that isn't distorted by
--            within-year seasonality the way a month-over-month figure is.
-- Expected insight: The headline number a board or investor update would
--            actually ask for -- "how does this year compare to last year."
WITH yearly_revenue AS (
    SELECT EXTRACT(YEAR FROM o.order_date)::int AS order_year, SUM(s.sales) AS revenue
    FROM sales s
    JOIN orders o ON s.order_id = o.order_id
    GROUP BY order_year
)
SELECT
    order_year,
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY order_year)) / NULLIF(LAG(revenue) OVER (ORDER BY order_year), 0), 2) AS yoy_growth_pct
FROM yearly_revenue
ORDER BY order_year;


-- Query 25: Year-over-year growth by category
-- Objective: Same YoY pattern as Query 24, but PARTITION BY category so
--            each category's year-over-year comparison is computed against
--            its own prior year, not the company-wide prior year.
-- Expected insight: Company-wide YoY growth (Query 24) can mask a
--            declining category being propped up by a growing one -- this
--            is the query that would actually catch that.
WITH yearly_category_revenue AS (
    SELECT
        EXTRACT(YEAR FROM o.order_date)::int AS order_year,
        c.category_name,
        SUM(s.sales) AS revenue
    FROM sales s
    JOIN orders o           ON s.order_id = o.order_id
    JOIN products p         ON s.product_id = p.product_id
    JOIN sub_categories sc  ON p.subcategory_id = sc.subcategory_id
    JOIN categories c       ON sc.category_id = c.category_id
    GROUP BY order_year, c.category_name
)
SELECT
    order_year,
    category_name,
    ROUND(revenue::numeric, 2) AS revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (PARTITION BY category_name ORDER BY order_year))
        / NULLIF(LAG(revenue) OVER (PARTITION BY category_name ORDER BY order_year), 0), 2) AS yoy_growth_pct
FROM yearly_category_revenue
ORDER BY category_name, order_year;


-- ============================================================================
-- SECTION 8: CUSTOMER ANALYSIS
-- ============================================================================

-- Query 26: Segment-level performance summary
-- Objective: One row per customer segment with the full KPI set --
--            revenue, profit, margin, order count, and AOV -- for a
--            side-by-side segment comparison.
-- Expected insight: If Home Office has the highest margin but the fewest
--            orders, it may be an under-marketed segment worth expanding
--            into, rather than one to deprioritize because of its smaller
--            current size.
SELECT
    c.segment,
    COUNT(DISTINCT o.order_id)        AS order_count,
    ROUND(SUM(s.sales)::numeric, 2)   AS revenue,
    ROUND(SUM(s.profit)::numeric, 2)  AS profit,
    ROUND((SUM(s.profit) / NULLIF(SUM(s.sales), 0) * 100)::numeric, 2) AS margin_pct,
    ROUND((SUM(s.sales) / NULLIF(COUNT(DISTINCT o.order_id), 0))::numeric, 2) AS avg_order_value
FROM sales s
JOIN orders o    ON s.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
GROUP BY c.segment
ORDER BY revenue DESC;


-- Query 27: Top 10 customers by lifetime revenue
-- Objective: The single highest-value customer list, with segment and
--            primary region attached for context on who they are.
-- Expected insight: A concrete named list for account management --
--            not just "the Corporate segment matters" but exactly which
--            ten relationships matter most.
SELECT
    c.customer_name,
    c.segment,
    r.region_name AS most_recent_region,
    ROUND(SUM(s.sales)::numeric, 2) AS lifetime_revenue,
    COUNT(DISTINCT o.order_id) AS order_count
FROM sales s
JOIN orders o     ON s.order_id = o.order_id
JOIN customers c  ON o.customer_id = c.customer_id
JOIN locations l  ON o.location_id = l.location_id
JOIN regions r    ON l.region_id = r.region_id
GROUP BY c.customer_id, c.customer_name, c.segment, r.region_name
ORDER BY lifetime_revenue DESC
LIMIT 10;


-- Query 28: New vs. returning customers, by month
-- Objective: For every order, determine (via a CTE marking each
--            customer's first-ever order date) whether it came from a
--            brand-new customer or a returning one, then aggregate by month.
-- Expected insight: A rising share of revenue from returning customers is
--            a healthier growth signal than the same revenue growth coming
--            entirely from new-customer acquisition, which is typically
--            more expensive to sustain.
WITH first_orders AS (
    SELECT customer_id, MIN(order_date) AS first_order_date
    FROM orders
    GROUP BY customer_id
)
SELECT
    DATE_TRUNC('month', o.order_date) AS order_month,
    CASE WHEN o.order_date = fo.first_order_date THEN 'New' ELSE 'Returning' END AS customer_type,
    COUNT(DISTINCT o.order_id) AS order_count,
    ROUND(SUM(s.sales)::numeric, 2) AS revenue
FROM orders o
JOIN first_orders fo ON o.customer_id = fo.customer_id
JOIN sales s          ON s.order_id = o.order_id
GROUP BY order_month, customer_type
ORDER BY order_month, customer_type;


-- ============================================================================
-- SECTION 9: PRODUCT ANALYSIS
-- ============================================================================

-- Query 29: Highest-margin products (with a minimum revenue floor)
-- Objective: Top 10 products by average margin, restricted with HAVING to
--            products with at least $500 in total revenue -- otherwise a
--            product with two lucky sales can top the list misleadingly.
-- Expected insight: These products deserve more shelf space / ad spend per
--            dollar invested than the portfolio average.
SELECT
    p.product_name,
    ROUND(SUM(s.sales)::numeric, 2)  AS revenue,
    ROUND(AVG(s.profit / NULLIF(s.sales,0)) * 100, 2) AS avg_margin_pct
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
HAVING SUM(s.sales) >= 500
ORDER BY avg_margin_pct DESC
LIMIT 10;


-- Query 30: Worst 10 products by total profit
-- Objective: The mirror image of Query 29 -- products actively destroying
--            the most profit in absolute dollar terms.
-- Expected insight: Candidates for re-pricing, bundling, or discontinuation
--            -- especially any that also appear in Query 7's "broad reach"
--            list, meaning they're popular AND unprofitable.
SELECT
    p.product_name,
    ROUND(SUM(s.sales)::numeric, 2)  AS revenue,
    ROUND(SUM(s.profit)::numeric, 2) AS total_profit
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
ORDER BY total_profit ASC
LIMIT 10;


-- Query 31: Sub-category share of category revenue
-- Objective: What fraction of each category's revenue each sub-category
--            within it represents.
-- Expected insight: A category whose revenue is dominated by a single
--            sub-category has concentration risk -- a downturn in that one
--            sub-category would disproportionately hurt the whole category.
SELECT
    c.category_name,
    sc.subcategory_name,
    ROUND(SUM(s.sales)::numeric, 2) AS revenue,
    ROUND(100.0 * SUM(s.sales) / SUM(SUM(s.sales)) OVER (PARTITION BY c.category_name), 2) AS pct_of_category_revenue
FROM sales s
JOIN products p        ON s.product_id = p.product_id
JOIN sub_categories sc ON p.subcategory_id = sc.subcategory_id
JOIN categories c      ON sc.category_id = c.category_id
GROUP BY c.category_name, sc.subcategory_name
ORDER BY c.category_name, revenue DESC;


-- ============================================================================
-- SECTION 10: REGION ANALYSIS
-- ============================================================================

-- Query 32: Region x category revenue matrix
-- Objective: A pivot-style table -- one row per region, one column per
--            category -- built with conditional aggregation
--            (SUM(CASE WHEN ...)), Postgres' standard pivot pattern in the
--            absence of a native PIVOT clause.
-- Expected insight: Surfaces region/category combinations that are
--            unusually strong or completely absent -- e.g. a region with
--            solid Technology revenue but almost no Furniture revenue,
--            worth investigating as either a market-fit gap or an
--            assortment gap.
SELECT
    r.region_name,
    ROUND(SUM(CASE WHEN c.category_name = 'Technology'      THEN s.sales ELSE 0 END)::numeric, 2) AS technology_revenue,
    ROUND(SUM(CASE WHEN c.category_name = 'Furniture'       THEN s.sales ELSE 0 END)::numeric, 2) AS furniture_revenue,
    ROUND(SUM(CASE WHEN c.category_name = 'Office Supplies' THEN s.sales ELSE 0 END)::numeric, 2) AS office_supplies_revenue
FROM sales s
JOIN orders o           ON s.order_id = o.order_id
JOIN locations l         ON o.location_id = l.location_id
JOIN regions r           ON l.region_id = r.region_id
JOIN products p          ON s.product_id = p.product_id
JOIN sub_categories sc   ON p.subcategory_id = sc.subcategory_id
JOIN categories c        ON sc.category_id = c.category_id
GROUP BY r.region_name
ORDER BY r.region_name;


-- Query 33: Best-performing state within each region
-- Objective: RANK() states by revenue, partitioned by region, to find the
--            single best-performing state inside each region rather than
--            a company-wide state ranking that would be dominated by
--            whichever region is largest overall.
-- Expected insight: A "lead state" per region -- useful for deciding where
--            a regional field team or local marketing budget should be
--            anchored first.
WITH state_revenue AS (
    SELECT
        r.region_name,
        l.state,
        SUM(s.sales) AS revenue,
        RANK() OVER (PARTITION BY r.region_name ORDER BY SUM(s.sales) DESC) AS state_rank
    FROM sales s
    JOIN orders o    ON s.order_id = o.order_id
    JOIN locations l ON o.location_id = l.location_id
    JOIN regions r   ON l.region_id = r.region_id
    GROUP BY r.region_name, l.state
)
SELECT region_name, state, ROUND(revenue::numeric, 2) AS revenue
FROM state_revenue
WHERE state_rank = 1
ORDER BY revenue DESC;


-- ============================================================================
-- SECTION 11: PROFITABILITY ANALYSIS
-- ============================================================================

-- Query 34: Profitability by shipping mode
-- Objective: Compare average margin across shipping modes -- does
--            expedited shipping (Same Day / First Class) come at a margin
--            cost, or is it margin-neutral?
-- Expected insight: If faster shipping modes show meaningfully lower
--            margin, that's a case for either pricing shipping speed as a
--            paid upgrade or accepting it as a deliberate loss-leader for
--            customer experience -- but it should be a deliberate choice,
--            not an undetected one.
SELECT
    o.ship_mode,
    COUNT(DISTINCT o.order_id) AS order_count,
    ROUND(SUM(s.sales)::numeric, 2)  AS revenue,
    ROUND((SUM(s.profit) / NULLIF(SUM(s.sales), 0) * 100)::numeric, 2) AS margin_pct
FROM sales s
JOIN orders o ON s.order_id = o.order_id
GROUP BY o.ship_mode
ORDER BY margin_pct DESC;


-- Query 35: Profit concentration (Pareto check)
-- Objective: Using a CTE to rank products by profit and compute a running
--            share of total profit, find what fraction of products account
--            for 80% of total profit -- the classic "80/20" check.
-- Expected insight: If a small fraction of products generate the large
--            majority of profit, the business is more concentrated (and
--            more exposed to losing any one of those products) than the
--            product catalog's size alone would suggest.
WITH product_profit AS (
    SELECT
        p.product_name,
        SUM(s.profit) AS total_profit
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_name
    HAVING SUM(s.profit) > 0
),
ranked AS (
    SELECT
        product_name,
        total_profit,
        SUM(total_profit) OVER (ORDER BY total_profit DESC) AS running_profit,
        SUM(total_profit) OVER ()                            AS total_profit_all_products,
        ROW_NUMBER() OVER (ORDER BY total_profit DESC)        AS product_rank,
        COUNT(*) OVER ()                                      AS total_product_count
    FROM product_profit
)
SELECT
    product_rank,
    total_product_count,
    ROUND(100.0 * product_rank / total_product_count, 1) AS pct_of_products_so_far,
    ROUND(100.0 * running_profit / total_profit_all_products, 1) AS pct_of_profit_so_far
FROM ranked
WHERE running_profit >= 0.8 * total_profit_all_products
ORDER BY product_rank
LIMIT 1;


-- ============================================================================
-- SECTION 12: RFM ANALYSIS (Recency, Frequency, Monetary)
-- ============================================================================

-- Query 36: RFM component calculation
-- Objective: Compute the three raw RFM inputs per customer in one query --
--            days since last order, number of distinct orders, and total
--            spend -- as the foundation the scoring query below builds on.
-- Expected insight: The raw building blocks for any RFM-based segmentation,
--            independent of how the scoring thresholds are eventually set.
WITH customer_rfm_raw AS (
    SELECT
        c.customer_id,
        c.customer_name,
        (SELECT MAX(order_date) FROM orders) - MAX(o.order_date) AS recency_days,
        COUNT(DISTINCT o.order_id)                                AS frequency,
        SUM(s.sales)                                              AS monetary
    FROM sales s
    JOIN orders o    ON s.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_name
)
SELECT customer_name, recency_days, frequency, ROUND(monetary::numeric, 2) AS monetary
FROM customer_rfm_raw
ORDER BY monetary DESC;


-- Query 37: RFM scoring (1-5 scale per component)
-- Objective: NTILE(5) each of recency, frequency, and monetary into
--            quintiles, then combine into a single 3-digit RFM score
--            (e.g. '555' = best possible customer).
-- Expected insight: A standardized score that's directly comparable across
--            customers and stable as a reporting metric over time, rather
--            than three separate numbers a stakeholder has to mentally combine.
WITH customer_rfm_raw AS (
    SELECT
        c.customer_id,
        c.customer_name,
        (SELECT MAX(order_date) FROM orders) - MAX(o.order_date) AS recency_days,
        COUNT(DISTINCT o.order_id)                                AS frequency,
        SUM(s.sales)                                              AS monetary
    FROM sales s
    JOIN orders o    ON s.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_name
),
scored AS (
    SELECT
        customer_name,
        recency_days, frequency, monetary,
        -- lower recency_days is better, so the quintile order is ascending
        NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
        NTILE(5) OVER (ORDER BY frequency DESC)     AS f_score,
        NTILE(5) OVER (ORDER BY monetary DESC)      AS m_score
    FROM customer_rfm_raw
)
SELECT
    customer_name,
    recency_days, frequency, ROUND(monetary::numeric, 2) AS monetary,
    r_score, f_score, m_score,
    (r_score::text || f_score::text || m_score::text) AS rfm_score
FROM scored
ORDER BY r_score DESC, f_score DESC, m_score DESC;


-- Query 38: RFM-based customer segments
-- Objective: Translate the numeric RFM scores from Query 37 into named,
--            actionable segments using CASE WHEN -- the step that turns a
--            score into something a marketing team can actually target.
-- Expected insight: A ready-to-export campaign list -- e.g. every
--            "At Risk" customer (historically strong, recently quiet) is
--            exactly who a win-back campaign should target first.
WITH customer_rfm_raw AS (
    SELECT
        c.customer_id,
        c.customer_name,
        (SELECT MAX(order_date) FROM orders) - MAX(o.order_date) AS recency_days,
        COUNT(DISTINCT o.order_id)                                AS frequency,
        SUM(s.sales)                                              AS monetary
    FROM sales s
    JOIN orders o    ON s.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.customer_id, c.customer_name
),
scored AS (
    SELECT
        customer_name,
        NTILE(5) OVER (ORDER BY recency_days ASC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency DESC)    AS f_score,
        NTILE(5) OVER (ORDER BY monetary DESC)     AS m_score
    FROM customer_rfm_raw
)
SELECT
    customer_name,
    r_score, f_score, m_score,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score <= 2 AND f_score >= 4 AND m_score >= 4 THEN 'At Risk'
        WHEN r_score >= 4 AND f_score <= 2                  THEN 'New / Promising'
        WHEN r_score <= 2 AND f_score <= 2 AND m_score <= 2  THEN 'Lost'
        ELSE 'Needs Attention'
    END AS rfm_segment
FROM scored
ORDER BY rfm_segment, r_score DESC;


-- ============================================================================
-- SECTION 13: INVENTORY-ORIENTED INSIGHTS
-- ============================================================================
-- The Superstore dataset has no stock-on-hand table, so "inventory" here is
-- read the way a demand-planning analyst would in its absence: sales
-- velocity and recency as a proxy for which products need active
-- replenishment attention vs. which are effectively dormant.

-- Query 39: Fast-moving products (highest sales velocity)
-- Objective: Total units sold per month of a product's active selling
--            window -- units-per-month is a fairer "how fast does this
--            move" measure than raw total units for products with
--            different amounts of history in the dataset.
-- Expected insight: The products most worth protecting against stockouts
--            -- demand for them is both high and consistent.
SELECT
    p.product_name,
    SUM(s.quantity) AS total_units_sold,
    (MAX(o.order_date) - MIN(o.order_date)) AS active_days,
    ROUND(SUM(s.quantity) / NULLIF(GREATEST(MAX(o.order_date) - MIN(o.order_date), 1) / 30.0, 0), 2) AS units_per_month
FROM sales s
JOIN orders o   ON s.order_id = o.order_id
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
HAVING SUM(s.quantity) >= 5
ORDER BY units_per_month DESC
LIMIT 15;


-- Query 40: Slow-moving / dormant products
-- Objective: Products whose most recent sale is furthest in the past
--            relative to the dataset's overall latest order date --
--            a proxy for inventory that has effectively stopped turning.
-- Expected insight: Candidates for a clearance markdown or delisting --
--            capital tied up in a product that hasn't sold recently is
--            capital not available for a fast-moving one.
SELECT
    p.product_name,
    MAX(o.order_date) AS last_sold_date,
    (SELECT MAX(order_date) FROM orders) - MAX(o.order_date) AS days_since_last_sale,
    SUM(s.quantity) AS total_units_sold
FROM sales s
JOIN orders o   ON s.order_id = o.order_id
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
ORDER BY days_since_last_sale DESC
LIMIT 15;


-- Query 41: Discount dependency by product
-- Objective: For each product, the share of its sales made at a nonzero
--            discount, alongside its average margin -- do heavily
--            discounted products still hold acceptable margin, or does
--            demand for them only exist because of the discount?
-- Expected insight: A product sold almost exclusively at a discount with a
--            thin resulting margin is effectively mispriced at full price
--            -- worth a base-price review rather than continuing to rely
--            on discounting to move it.
SELECT
    p.product_name,
    COUNT(*)                                                    AS total_sales,
    SUM(CASE WHEN s.discount > 0 THEN 1 ELSE 0 END)             AS discounted_sales,
    ROUND(100.0 * SUM(CASE WHEN s.discount > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_sold_at_discount,
    ROUND(AVG(s.profit / NULLIF(s.sales,0)) * 100, 2)           AS avg_margin_pct
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_name
HAVING COUNT(*) >= 5
ORDER BY pct_sold_at_discount DESC
LIMIT 15;
