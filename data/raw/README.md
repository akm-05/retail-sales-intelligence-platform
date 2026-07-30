# data/raw/

Place the Sample Superstore dataset here as `superstore.csv`.

The raw CSV isn't committed to this repository (see `.gitignore`) --
download it separately (widely available, e.g. via Kaggle's "Superstore
Sales Dataset") and drop it in this folder before running:

```bash
python -m src.run_pipeline
```

Expected columns (matches the standard Superstore export): Row ID, Order
ID, Order Date, Ship Date, Ship Mode, Customer ID, Customer Name, Segment,
Country, City, State, Postal Code, Region, Product ID, Category,
Sub-Category, Product Name, Sales, Quantity, Discount, Profit.
