"""
Column detection and data cleaning utilities for user-uploaded files.
No Flask imports — pure Python, fully testable.
"""
import pandas as pd

COLUMN_ALIASES: dict[str, list[str]] = {
    'Customer ID': [
        'customer id', 'customerid', 'customer_id', 'cust_id', 'custid',
        'client id', 'client_id', 'user id', 'user_id', 'buyer id', 'buyer_id',
        'account id', 'account_id', 'member id', 'member_id', 'contact id',
    ],
    'Invoice': [
        'invoice', 'invoice_id', 'invoiceid', 'invoice no', 'invoice_no',
        'invoice number', 'order id', 'order_id', 'orderid', 'order no',
        'order number', 'order_number', 'transaction id', 'transaction_id',
        'txn_id', 'txid', 'receipt id', 'receipt_id', 'sale id', 'sale_id',
        'ref', 'reference',
    ],
    'Description': [
        'description', 'product description', 'product_description',
        'item description', 'item_description', 'product name', 'product_name',
        'productname', 'item name', 'item_name', 'product', 'item', 'sku name',
        'sku_name', 'article', 'goods', 'name', 'product title', 'title',
    ],
    'Quantity': [
        'quantity', 'qty', 'units', 'count', 'volume', 'number of items',
        'items', 'num_units', 'ordered_qty', 'order_qty', 'pieces',
    ],
    'Price': [
        'price', 'unit price', 'unit_price', 'unitprice', 'price per unit',
        'rate', 'cost', 'sale price', 'selling price', 'item price',
        'unit cost', 'unit_cost', 'list price', 'retail price',
    ],
    'InvoiceDate': [
        'invoicedate', 'invoice date', 'invoice_date', 'date', 'order date',
        'order_date', 'orderdate', 'transaction date', 'transaction_date',
        'purchase date', 'purchase_date', 'sale date', 'sale_date',
        'created at', 'created_at', 'timestamp', 'datetime', 'time',
    ],
    'Country': [
        'country', 'country name', 'country_name', 'nation', 'region',
        'territory', 'market', 'geography', 'location', 'billing country',
        'billing_country', 'ship country', 'ship_country',
    ],
}

REQUIRED_COLUMNS = ['Customer ID', 'Invoice', 'Description', 'Quantity', 'Price', 'InvoiceDate']
ALL_COLUMNS = list(COLUMN_ALIASES.keys())


def detect_column_mapping(df_columns: list[str]) -> dict[str, str | None]:
    """
    Auto-map uploaded DataFrame columns to the 7 canonical schema columns.
    Returns {canonical: matched_col_name | None}.
    """
    normalized = {col.lower().strip(): col for col in df_columns}
    mapping: dict[str, str | None] = {}
    used: set[str] = set()

    for canonical, aliases in COLUMN_ALIASES.items():
        found = None
        # Exact canonical name first (case-insensitive)
        if canonical.lower() in normalized:
            candidate = normalized[canonical.lower()]
            if candidate not in used:
                found = candidate
        # Then aliases
        if not found:
            for alias in aliases:
                if alias in normalized:
                    candidate = normalized[alias]
                    if candidate not in used:
                        found = candidate
                        break
        mapping[canonical] = found
        if found:
            used.add(found)

    return mapping


def mapping_is_complete(mapping: dict[str, str | None]) -> bool:
    """Return True only if all required columns have a mapping."""
    return all(mapping.get(col) is not None for col in REQUIRED_COLUMNS)


def apply_mapping_and_clean(
    df: pd.DataFrame,
    mapping: dict[str, str | None],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Rename columns per mapping then apply DataAgent-style cleaning.

    Returns:
        (cleaned_df, list_of_warning_strings)

    Raises:
        ValueError if a required mapped column is missing from df.
    """
    warnings: list[str] = []

    # Validate required mapped columns exist
    for canonical in REQUIRED_COLUMNS:
        src = mapping.get(canonical)
        if not src:
            raise ValueError(
                f"Required column '{canonical}' has no mapping. "
                f"Please assign one of your columns to it."
            )
        if src not in df.columns:
            raise ValueError(
                f"Mapped column '{src}' → '{canonical}' not found in uploaded data."
            )

    # Build rename dict — skip if source already matches canonical name
    rename_map = {v: k for k, v in mapping.items() if v and v != k and v in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    rows_before = len(df)

    # Deduplicate
    df = df.drop_duplicates()

    # Coerce numeric types before filtering
    for col in ['Quantity', 'Price']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows without Customer ID
    before = len(df)
    df = df.dropna(subset=['Customer ID'])
    removed = before - len(df)
    if removed:
        warnings.append(f"Removed {removed:,} rows with missing Customer ID.")

    # Drop rows without Description
    before = len(df)
    df = df.dropna(subset=['Description'])
    removed = before - len(df)
    if removed:
        warnings.append(f"Removed {removed:,} rows with missing Description.")

    # Drop zero/negative Price (negative Quantity = returns, those stay)
    before = len(df)
    df = df[df['Price'] > 0]
    removed = before - len(df)
    if removed:
        warnings.append(f"Removed {removed:,} rows with zero or negative Price.")

    # Parse InvoiceDate
    if 'InvoiceDate' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        nat_count = int(df['InvoiceDate'].isna().sum())
        if nat_count:
            warnings.append(
                f"{nat_count:,} rows have unparseable dates and will be excluded "
                f"from date-based analysis."
            )

    rows_after = len(df)
    if rows_before > rows_after:
        warnings.insert(
            0,
            f"Cleaned: kept {rows_after:,} of {rows_before:,} rows "
            f"({rows_before - rows_after:,} removed)."
        )

    df = df.reset_index(drop=True)
    return df, warnings
