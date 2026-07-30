-- ============================================================================
-- schema.sql
--
-- 3NF-normalized schema for the Retail Sales Intelligence & Demand
-- Forecasting Platform. Eight tables, matching the exact shapes produced by
-- src/preprocessing.py build_normalized_tables(): regions, categories,
-- locations, sub_categories, customers, products, orders, sales.
--
-- Table order below is FK-dependency order -- every table is declared only
-- after every table it references, matching TABLE_LOAD_ORDER in
-- src/sql_connector.py.
--
-- A note on how this file is executed: src/sql_connector.py's run_ddl()
-- splits this file on every literal semicolon character and discards any
-- resulting chunk whose stripped text starts with a comment marker. That
-- has two consequences for how this file must be written -- no comment may
-- sit directly in front of the statement it describes with no semicolon in
-- between, and no comment anywhere in the file may itself contain a
-- semicolon character. The SELECT 1 statement directly below exists only
-- to absorb this header block, so the loader does not merge it into --
-- and silently discard -- the first real CREATE TABLE statement.
-- ============================================================================
SELECT 1;

DROP TABLE IF EXISTS sales CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS sub_categories CASCADE;
DROP TABLE IF EXISTS locations CASCADE;
DROP TABLE IF EXISTS categories CASCADE;
DROP TABLE IF EXISTS regions CASCADE;

CREATE TABLE regions (
    region_id SERIAL PRIMARY KEY,
    region_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE categories (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE locations (
    location_id SERIAL PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    postal_code VARCHAR(20) NOT NULL,
    region_id INT NOT NULL REFERENCES regions (region_id),
    CONSTRAINT uq_locations_city_state_postal UNIQUE (city, state, postal_code)
);

CREATE TABLE sub_categories (
    subcategory_id SERIAL PRIMARY KEY,
    subcategory_name VARCHAR(50) NOT NULL,
    category_id INT NOT NULL REFERENCES categories (category_id),
    CONSTRAINT uq_sub_categories_name_category UNIQUE (subcategory_name, category_id)
);

CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    customer_name VARCHAR(150) NOT NULL,
    segment VARCHAR(30) NOT NULL
);

CREATE TABLE products (
    product_id VARCHAR(30) PRIMARY KEY,
    product_name VARCHAR(300) NOT NULL,
    subcategory_id INT NOT NULL REFERENCES sub_categories (subcategory_id)
);

CREATE TABLE orders (
    order_id VARCHAR(20) PRIMARY KEY,
    order_date DATE NOT NULL,
    ship_date DATE NOT NULL,
    ship_mode VARCHAR(30) NOT NULL,
    customer_id VARCHAR(20) NOT NULL REFERENCES customers (customer_id),
    location_id INT NOT NULL REFERENCES locations (location_id),
    CONSTRAINT chk_orders_ship_after_order CHECK (ship_date >= order_date)
);

CREATE TABLE sales (
    sales_id SERIAL PRIMARY KEY,
    order_id VARCHAR(20) NOT NULL REFERENCES orders (order_id),
    product_id VARCHAR(30) NOT NULL REFERENCES products (product_id),
    sales NUMERIC(12, 4) NOT NULL CHECK (sales >= 0),
    quantity INT NOT NULL CHECK (quantity > 0),
    discount NUMERIC(5, 4) NOT NULL CHECK (discount >= 0 AND discount <= 1),
    profit NUMERIC(12, 4) NOT NULL
);

CREATE INDEX idx_locations_region_id ON locations (region_id);
CREATE INDEX idx_sub_categories_category_id ON sub_categories (category_id);
CREATE INDEX idx_products_subcategory_id ON products (subcategory_id);
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
CREATE INDEX idx_orders_location_id ON orders (location_id);
CREATE INDEX idx_orders_order_date ON orders (order_date);
CREATE INDEX idx_sales_order_id ON sales (order_id);
CREATE INDEX idx_sales_product_id ON sales (product_id);
