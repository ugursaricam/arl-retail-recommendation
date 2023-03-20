## The dataset named "Online Retail II" includes the sales of an UK-based online store between 01/12/2009 - 09/12/2011.
The "online_retail_II.xlsx" dataset contains transactional data of a UK-based online retailer that sells various types of gifts. The data covers the period of 01/12/2009 to 09/12/2011 and includes customer ID, transaction date, product ID, product description, quantity, and price. The dataset has 8 sheets, but we'll mainly focus on the "Online Retail" sheet, which contains more than 1 million rows of transactional data. The goal of using this dataset is to demonstrate the use of apriori algorithm and association rules to create a product recommendation system.

dataset: https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

Variables
* **Invoice:** Invoice number. The unique number of each transaction, namely the invoice. If it starts with C, it shows the canceled invoice
* **StockCode:** A 5-digit integral number uniquely assigned to each distinct product.
* **Description:** Product description
* **Quantity:** The quantities of each product (item) per transaction.
* **InvoiceDate:** The day and time when a transaction was generated.
* **UnitPrice:** Product price (in GBP)
* **CustomerID:** Unique customer number
* **Country:** The name of the country where a customer resides.