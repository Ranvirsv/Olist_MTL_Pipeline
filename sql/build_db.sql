-- 1. Ingest All 8 Raw CSVs into DuckDB Tables
CREATE OR REPLACE TABLE customers AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_customers_dataset.csv');
CREATE OR REPLACE TABLE orders AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_orders_dataset.csv');
CREATE OR REPLACE TABLE reviews AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_order_reviews_dataset.csv');
CREATE OR REPLACE TABLE items AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_order_items_dataset.csv');
CREATE OR REPLACE TABLE products AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_products_dataset.csv');
CREATE OR REPLACE TABLE sellers AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_sellers_dataset.csv');
CREATE OR REPLACE TABLE payments AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_order_payments_dataset.csv');
CREATE OR REPLACE TABLE geolocation AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_geolocation_dataset.csv');

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
)

SELECT 
    o.order_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state,
    o.order_purchase_timestamp,
    
    -- TARGET A & B
    date_diff('day', CAST(o.order_purchase_timestamp AS DATE), CAST(o.order_delivered_customer_date AS DATE)) AS delivery_days,
    r.review_score,
    r.review_comment_message,

    -- FEATURES
    i.price,
    i.freight_value,
    p.product_weight_g,
    p.product_category_name,
    s.seller_state,
    pay.total_payment_value,
    pay.max_installments,
    
    -- LAT/LNG FOR DISTANCE CALCULATION
    c_geo.lat AS customer_lat,
    c_geo.lng AS customer_lng,
    s_geo.lat AS seller_lat,
    s_geo.lng AS seller_lng

FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN reviews r ON o.order_id = r.order_id
LEFT JOIN items i ON o.order_id = i.order_id
LEFT JOIN products p ON i.product_id = p.product_id
LEFT JOIN sellers s ON i.seller_id = s.seller_id
LEFT JOIN aggregated_payments pay ON o.order_id = pay.order_id
LEFT JOIN distinct_geo c_geo ON c.customer_zip_code_prefix = c_geo.zip_code
LEFT JOIN distinct_geo s_geo ON s.seller_zip_code_prefix = s_geo.zip_code
WHERE o.order_status = 'delivered';