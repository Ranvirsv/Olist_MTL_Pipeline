-- 1. Ingest All 8 Raw CSVs into DuckDB Tables
CREATE OR REPLACE TABLE customers AS SELECT * FROM read_csv_auto('./data/raw/olist_customers_dataset.csv');
CREATE OR REPLACE TABLE orders AS SELECT * FROM read_csv_auto('./data/raw/olist_orders_dataset.csv');
CREATE OR REPLACE TABLE reviews AS SELECT * FROM read_csv_auto('./data/raw/olist_order_reviews_dataset.csv');
CREATE OR REPLACE TABLE items AS SELECT * FROM read_csv_auto('./data/raw/olist_order_items_dataset.csv');
CREATE OR REPLACE TABLE products AS SELECT * FROM read_csv_auto('./data/raw/olist_products_dataset.csv');
CREATE OR REPLACE TABLE sellers AS SELECT * FROM read_csv_auto('./data/raw/olist_sellers_dataset.csv');
CREATE OR REPLACE TABLE payments AS SELECT * FROM read_csv_auto('./data/raw/olist_order_payments_dataset.csv');
CREATE OR REPLACE TABLE geolocation AS SELECT * FROM read_csv_auto('./data/raw/olist_geolocation_dataset.csv');

-- 2. Build the Unified Analytical View
CREATE OR REPLACE VIEW analytical_base_table AS 
-- CTE 1: Aggregate Payments (An order can have multiple payments, we need the sum)
WITH aggregated_payments AS (
    SELECT 
        order_id, 
        SUM(payment_value) AS total_payment_value,
        MAX(payment_installments) AS max_installments
    FROM payments
    GROUP BY order_id
),
-- CTE 2: Aggregate Geolocation (Olist has duplicate lat/lngs for the same zip code)
distinct_geo AS (
    SELECT 
        geolocation_zip_code_prefix AS zip_code,
        AVG(geolocation_lat) AS lat,
        AVG(geolocation_lng) AS lng
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
),
-- CTE 3: Aggrigated items
aggregated_items AS (
    SELECT
        order_id,
        SUM(price) AS total_price,
        SUM(freight_value) AS total_freight_value,
        COUNT(*) AS num_items,
        MIN(seller_id)     AS seller_id,
        MIN(product_id)    AS product_id
    FROM items
    GROUP BY order_id
),
-- CTE 4: Seller historical volume (total orders fulfilled)
seller_stats AS (
    SELECT
        s.seller_id,
        COUNT(DISTINCT i.order_id) AS seller_order_count,
    FROM items i
    JOIN sellers s ON i.seller_id = s.seller_id
    GROUP BY s.seller_id
),
-- CTE 5: Seller historical avg review score
seller_reviews AS (
    SELECT
        i.seller_id,
        AVG(r.review_score) AS seller_avg_review
    FROM items i
    JOIN reviews r ON i.order_id = r.order_id
    GROUP BY i.seller_id
)

SELECT
    DISTINCT(o.order_id),
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_purchase_timestamp,

    -- TARGET A & B
    date_diff('day', CAST(o.order_purchase_timestamp AS DATE), CAST(o.order_delivered_customer_date AS DATE)) AS delivery_days,
    r.review_score,
    r.review_comment_message,

    -- FEATURES (original)
    i.total_price,
    i.total_freight_value,
    p.product_weight_g,
    p.product_category_name,
    s.seller_state,
    pay.total_payment_value,
    pay.max_installments,

    -- FEATURES (new: delivery lateness — negative = early, positive = late)
    date_diff('day', CAST(o.order_estimated_delivery_date AS DATE), CAST(o.order_delivered_customer_date AS DATE)) AS delivery_lateness_days,

    -- FEATURES (new: product detail)
    p.product_description_lenght AS product_description_length,
    p.product_photos_qty,
    COALESCE(p.product_length_cm * p.product_height_cm * p.product_width_cm, NULL) AS product_volume_cm3,

    -- FEATURES (new: order complexity)
    i.num_items,

    -- FEATURES (new: seller reputation)
    ss.seller_order_count,
    sr.seller_avg_review,

    -- LAT/LNG FOR DISTANCE CALCULATION
    c_geo.lat AS customer_lat,
    c_geo.lng AS customer_lng,
    s_geo.lat AS seller_lat,
    s_geo.lng AS seller_lng

FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN reviews r ON o.order_id = r.order_id
LEFT JOIN aggregated_items i ON o.order_id = i.order_id
LEFT JOIN products p ON i.product_id = p.product_id
LEFT JOIN sellers s ON i.seller_id = s.seller_id
LEFT JOIN aggregated_payments pay ON o.order_id = pay.order_id
LEFT JOIN seller_stats ss ON i.seller_id = ss.seller_id
LEFT JOIN seller_reviews sr ON i.seller_id = sr.seller_id
LEFT JOIN distinct_geo c_geo ON c.customer_zip_code_prefix = c_geo.zip_code
LEFT JOIN distinct_geo s_geo ON s.seller_zip_code_prefix = s_geo.zip_code
WHERE o.order_status = 'delivered';