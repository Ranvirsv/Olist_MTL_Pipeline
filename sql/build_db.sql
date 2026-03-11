CREATE OR REPLACE TABLE customers AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_customers_dataset.csv');
CREATE OR REPLACE TABLE orders AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_orders_dataset.csv');
CREATE OR REPLACE TABLE reviews AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_order_reviews_dataset.csv');
CREATE OR REPLACE TABLE items AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_order_items_dataset.csv');
CREATE OR REPLACE TABLE products AS SELECT * FROM read_csv_auto('/Users/ranvirsinghvirk/Eclipse/DS Practice/Olist_MTL_Pipeline/data/raw/olist_products_dataset.csv');

CREATE VIEW analytical_base_table AS
SELECT 
    o.order_id, 
    o.order_purchase_timestamp, 

    -- Get the days between order purchase and order delivery
    DATE_DIFF('days', CAST(o.order_purchase_timestamp AS DATE), CAST(o.order_delivered_customer_date AS DATE)) AS days_to_deliver,

    c.customer_unique_id, 
    c.customer_city, 
    c.customer_state,
    r.review_score,
    r.review_comment_message,
    i.price,
    i.freight_value,
    p.product_weight_g,
    p.product_category_name
FROM orders AS o
JOIN customers AS c ON o.customer_id = c.customer_id
JOIN items AS i ON o.order_id = i.order_id
JOIN products as p ON i.product_id = p.product_id
JOIN reviews AS r ON o.order_id = r.order_id
WHERE o.order_status = 'delivered'