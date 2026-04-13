"""
Prediction_Analyst.py — Pre-built forecasting and predictive analytics tools.

All methods use only pandas + numpy (no external ML libraries required).
Predictions are pattern-based on historical transaction data — no inventory,
cost, or marketing data is available in the dataset.
"""

import pandas as pd
import numpy as np


class PredictionAnalyst:
    def __init__(self, data_frame: pd.DataFrame):
        self.df = data_frame.copy()

        # ── Column presence flags (checked once; used everywhere) ─────────────
        self._has_invoice_date = "InvoiceDate" in self.df.columns
        self._has_customer_id  = "Customer ID" in self.df.columns
        self._has_invoice      = "Invoice" in self.df.columns
        self._has_description  = "Description" in self.df.columns
        self._has_quantity     = "Quantity" in self.df.columns
        self._has_price        = "Price" in self.df.columns

        # ── Revenue ───────────────────────────────────────────────────────────
        if self._has_quantity and self._has_price:
            self.df["Revenue"] = self.df["Quantity"] * self.df["Price"]
        self._has_revenue = "Revenue" in self.df.columns

        # ── Date parsing ──────────────────────────────────────────────────────
        if self._has_invoice_date:
            self.df["InvoiceDate"] = pd.to_datetime(
                self.df["InvoiceDate"], errors="coerce"
            )
            # Drop rows whose date could not be parsed (NaT) for date-sensitive ops
            self._dated_df = self.df.dropna(subset=["InvoiceDate"])
            self._reference_date: pd.Timestamp | None = (
                self._dated_df["InvoiceDate"].max()
                if not self._dated_df.empty
                else None
            )
        else:
            self._dated_df = self.df.iloc[0:0]  # empty frame, same schema
            self._reference_date = None

        self._has_valid_dates = (
            self._reference_date is not None and pd.notna(self._reference_date)
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _missing_col_error(self, col: str, as_list: bool = False):
        msg = {"error": f"Required column '{col}' is missing from the dataset."}
        return [msg] if as_list else msg

    def _no_date_error(self, as_list: bool = False):
        msg = {"error": "No valid InvoiceDate values found. All dates are missing or corrupt."}
        return [msg] if as_list else msg

    # ──────────────────────────────────────────────────────────────────────────
    # Churn risk
    # ──────────────────────────────────────────────────────────────────────────

    def get_churn_risk_summary(self, days_inactive: int = 90) -> dict:
        """Returns a high-level churn risk summary: how many customers haven't purchased
        in the last `days_inactive` days (relative to the dataset's last date).
        Use this for a quick 'how bad is churn?' overview before diving deeper.
        Args:
            days_inactive: Days without a purchase to classify a customer as at-risk (default 90).
        """
        if not self._has_invoice_date:
            return self._missing_col_error("InvoiceDate")
        if not self._has_customer_id:
            return self._missing_col_error("Customer ID")
        if not self._has_valid_dates:
            return self._no_date_error()

        last_purchase = self._dated_df.groupby("Customer ID")["InvoiceDate"].max()
        days_since = (self._reference_date - last_purchase).dt.days

        total    = len(days_since)
        at_risk  = int((days_since >= days_inactive).sum())
        healthy  = total - at_risk

        return {
            "reference_date":        str(self._reference_date.date()),
            "days_inactive_threshold": days_inactive,
            "total_customers":       total,
            "at_risk_customers":     at_risk,
            "healthy_customers":     healthy,
            "churn_risk_pct":        round(at_risk / total * 100, 1) if total > 0 else 0.0,
            "note": "Churn is measured relative to the dataset's last recorded date, not today.",
        }

    def get_at_risk_customers(self, days_inactive: int = 90, top_n: int = 15) -> list:
        """Returns the top N at-risk customers most likely to churn, sorted by total lifetime spend
        (highest-value customers first — most important to retain).
        Use this to build a targeted retention list.
        Args:
            days_inactive: Days without a purchase to classify a customer as at-risk (default 90).
            top_n: How many customers to return (default 15, max 50).
        """
        if not self._has_invoice_date:
            return [self._missing_col_error("InvoiceDate")]
        if not self._has_customer_id:
            return [self._missing_col_error("Customer ID")]
        if not self._has_valid_dates:
            return self._no_date_error(as_list=True)
        top_n = min(top_n, 50)

        last_purchase = self._dated_df.groupby("Customer ID")["InvoiceDate"].max()
        days_since    = (self._reference_date - last_purchase).dt.days
        at_risk_ids   = days_since[days_since >= days_inactive].index

        total_spend: pd.Series = pd.Series(dtype=float)
        if self._has_revenue and self._has_quantity:
            pos = self.df[self.df["Quantity"] > 0]
            total_spend = pos.groupby("Customer ID")["Revenue"].sum()

        result = []
        for cid in at_risk_ids:
            # cid may be float (e.g. 12345.0) — convert safely
            try:
                cid_display = int(cid)
            except (ValueError, TypeError):
                cid_display = cid
            result.append({
                "customer_id":              cid_display,
                "days_since_last_purchase": int(days_since[cid]),
                "total_spend_gbp":          round(float(total_spend.get(cid, 0.0)), 2),
                "last_purchase_date":       str(last_purchase[cid].date()),
            })

        result.sort(key=lambda x: x["total_spend_gbp"], reverse=True)
        return result[:top_n]

    # ──────────────────────────────────────────────────────────────────────────
    # Revenue forecasting
    # ──────────────────────────────────────────────────────────────────────────

    def get_revenue_forecast(self, horizon_months: int = 3) -> dict:
        """Forecasts monthly revenue for the next N months using linear regression on historical
        monthly revenue. Returns the predicted revenue for each future month and the trend slope.
        Use this for 'what will revenue look like next quarter?' questions.
        Args:
            horizon_months: How many months ahead to forecast (default 3, max 12).
        """
        if not self._has_invoice_date:
            return self._missing_col_error("InvoiceDate")
        if not self._has_revenue:
            return {"error": "Revenue could not be computed. Ensure both 'Price' and 'Quantity' columns exist in the dataset."}
        if not self._has_valid_dates:
            return self._no_date_error()
        horizon_months = min(horizon_months, 12)

        sales_df = self._dated_df[self._dated_df["Quantity"] > 0].copy() if self._has_quantity else self._dated_df.copy()
        monthly = (
            sales_df.set_index("InvoiceDate")
            .resample("ME")["Revenue"]
            .sum()
            .reset_index()
        )
        monthly.columns = ["month", "revenue"]
        monthly = monthly[monthly["revenue"] > 0]

        if len(monthly) < 3:
            return {"error": "Not enough monthly data to generate a forecast (need at least 3 months)."}

        # ── Outlier detection: warn if any month is > 5× the median ─────────
        # IQR-based detection fails when most values are identical (IQR=0),
        # so we use a median-ratio check which is robust to that edge case.
        median_rev = float(monthly["revenue"].median())
        outlier_warning = None
        if median_rev > 0:
            spike_mask     = monthly["revenue"] > 5 * median_rev
            outlier_months = monthly[spike_mask]
            if not outlier_months.empty:
                worst = float(outlier_months["revenue"].max())
                outlier_warning = (
                    f"WARNING: {len(outlier_months)} month(s) contain revenue spikes "
                    f"({worst:,.0f} £ vs median {median_rev:,.0f} £) that are >5× the typical "
                    "month and will heavily skew this linear forecast. "
                    "Treat projected values with caution."
                )

        # Numeric index for regression
        x = np.arange(len(monthly))
        y = monthly["revenue"].values
        slope, intercept = np.polyfit(x, y, 1)

        trend_direction = "upward" if slope > 0 else "downward" if slope < 0 else "flat"

        # Project forward
        last_month = monthly["month"].iloc[-1]
        forecasts: dict[str, float] = {}
        for i in range(1, horizon_months + 1):
            future_x   = len(monthly) - 1 + i
            predicted  = intercept + slope * future_x
            future_month = (last_month + pd.DateOffset(months=i)).to_period("M")
            forecasts[str(future_month)] = round(float(max(predicted, 0.0)), 2)

        last_actual  = round(float(monthly["revenue"].iloc[-1]), 2)
        avg_monthly  = round(float(monthly["revenue"].mean()), 2)

        result: dict = {
            "forecast":                   forecasts,
            "trend":                      trend_direction,
            "monthly_slope_gbp":          round(float(slope), 2),
            "last_actual_month":          str(monthly["month"].iloc[-1].to_period("M")),
            "last_actual_revenue_gbp":    last_actual,
            "historical_avg_monthly_gbp": avg_monthly,
            "method":                     "linear regression on monthly revenue",
            "warning": (
                "Linear regression assumes the current trend continues. "
                "Seasonal effects, holidays, or external shocks are not modelled."
            ),
        }
        if outlier_warning:
            result["outlier_warning"] = outlier_warning
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Product demand trends
    # ──────────────────────────────────────────────────────────────────────────

    def get_product_demand_trend(self, product_desc: str) -> dict:
        """Analyses whether demand for a specific product is growing, declining, or stable
        by fitting a linear trend to its monthly sales quantity.
        Use this when the user asks 'is [product] growing?' or 'what's the trend for [product]?'
        IMPORTANT: call search_products first if the product name might be approximate.
        Args:
            product_desc: Exact product description string (case-insensitive match applied).
        """
        if not self._has_description:
            return self._missing_col_error("Description")
        if not self._has_invoice_date:
            return self._missing_col_error("InvoiceDate")
        if not self._has_valid_dates:
            return self._no_date_error()

        mask       = self.df["Description"].str.lower() == product_desc.lower()
        qty_filter = self.df["Quantity"] > 0 if self._has_quantity else pd.Series(True, index=self.df.index)
        product_df = self._dated_df[
            mask.reindex(self._dated_df.index, fill_value=False) & qty_filter.reindex(self._dated_df.index, fill_value=False)
        ]

        if product_df.empty:
            return {"error": f"No sales data found for '{product_desc}'."}

        monthly_qty = (
            product_df.set_index("InvoiceDate")
            .resample("ME")["Quantity"]
            .sum()
        )

        if len(monthly_qty) < 2:
            return {
                "product": product_desc,
                "trend":   "insufficient data",
                "note":    "Only one month of data available — cannot determine trend.",
            }

        x = np.arange(len(monthly_qty))
        slope, _ = np.polyfit(x, monthly_qty.values, 1)

        if slope > 5:
            trend = "growing"
        elif slope < -5:
            trend = "declining"
        else:
            trend = "stable"

        monthly_dict = {
            str(k.to_period("M")): int(v) for k, v in monthly_qty.items()
        }

        return {
            "product":          product_desc,
            "trend":            trend,
            "monthly_slope_units": round(float(slope), 2),
            "total_units_sold": int(product_df["Quantity"].sum()),
            "months_of_data":   len(monthly_qty),
            "monthly_breakdown": monthly_dict,
        }

    def get_high_growth_products(self, lookback_months: int = 3, top_n: int = 5) -> list:
        """Finds the top N products with the highest quantity growth over the most recent
        `lookback_months` compared to the same number of months before that.
        Use this for 'what products are taking off?' or 'what should we stock more of?'
        Args:
            lookback_months: How many recent months to compare (default 3, max 6).
            top_n: How many products to return (default 5, max 20).
        """
        if not self._has_description:
            return [self._missing_col_error("Description")]
        if not self._has_revenue:
            return [{"error": "Revenue could not be computed. Ensure both 'Price' and 'Quantity' columns exist in the dataset."}]
        if not self._has_valid_dates:
            return self._no_date_error(as_list=True)
        lookback_months = min(lookback_months, 6)
        top_n = min(top_n, 20)

        sales_df     = self._dated_df[self._dated_df["Quantity"] > 0].copy() if self._has_quantity else self._dated_df.copy()
        cutoff       = self._reference_date - pd.DateOffset(months=lookback_months)
        prior_cutoff = cutoff - pd.DateOffset(months=lookback_months)

        recent = sales_df[sales_df["InvoiceDate"] >= cutoff]
        prior  = sales_df[(sales_df["InvoiceDate"] >= prior_cutoff) & (sales_df["InvoiceDate"] < cutoff)]

        if recent.empty or prior.empty:
            return [{"error": "Not enough historical data to calculate growth."}]

        recent_qty = recent.groupby("Description")["Quantity"].sum()
        prior_qty  = prior.groupby("Description")["Quantity"].sum()

        combined = pd.DataFrame({"recent": recent_qty, "prior": prior_qty}).dropna()
        combined = combined[combined["prior"] > 0]

        if combined.empty:
            return [{"error": "No products with data in both comparison windows."}]

        combined["growth_pct"] = ((combined["recent"] - combined["prior"]) / combined["prior"] * 100).round(1)

        top = combined.nlargest(top_n, "growth_pct")
        return [
            {
                "product":      desc,
                "recent_units": int(row["recent"]),
                "prior_units":  int(row["prior"]),
                "growth_pct":   float(row["growth_pct"]),
            }
            for desc, row in top.iterrows()
        ]

    def get_slow_movers(self, lookback_months: int = 3, top_n: int = 10) -> list:
        """Finds the top N products with the steepest quantity decline over the most recent
        `lookback_months` compared to the same number of months before that.
        Use this for 'what products are dying?' or 'what should we discount or discontinue?'
        Args:
            lookback_months: How many recent months to compare (default 3, max 6).
            top_n: How many products to return (default 10, max 20).
        """
        if not self._has_description:
            return [self._missing_col_error("Description")]
        if not self._has_revenue:
            return [{"error": "Revenue could not be computed. Ensure both 'Price' and 'Quantity' columns exist in the dataset."}]
        if not self._has_valid_dates:
            return self._no_date_error(as_list=True)
        lookback_months = min(lookback_months, 6)
        top_n = min(top_n, 20)

        sales_df     = self._dated_df[self._dated_df["Quantity"] > 0].copy() if self._has_quantity else self._dated_df.copy()
        cutoff       = self._reference_date - pd.DateOffset(months=lookback_months)
        prior_cutoff = cutoff - pd.DateOffset(months=lookback_months)

        recent = sales_df[sales_df["InvoiceDate"] >= cutoff]
        prior  = sales_df[(sales_df["InvoiceDate"] >= prior_cutoff) & (sales_df["InvoiceDate"] < cutoff)]

        if recent.empty or prior.empty:
            return [{"error": "Not enough historical data to identify slow movers."}]

        recent_qty = recent.groupby("Description")["Quantity"].sum()
        prior_qty  = prior.groupby("Description")["Quantity"].sum()

        combined = pd.DataFrame({"recent": recent_qty, "prior": prior_qty}).dropna()
        combined = combined[combined["prior"] > 0]

        if combined.empty:
            return [{"error": "No products with data in both comparison windows."}]

        combined["decline_pct"] = ((combined["prior"] - combined["recent"]) / combined["prior"] * 100).round(1)

        top = combined.nlargest(top_n, "decline_pct")
        return [
            {
                "product":      desc,
                "recent_units": int(row["recent"]),
                "prior_units":  int(row["prior"]),
                "decline_pct":  float(row["decline_pct"]),
            }
            for desc, row in top.iterrows()
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Customer behaviour predictions
    # ──────────────────────────────────────────────────────────────────────────

    def get_repeat_purchase_probability(self) -> dict:
        """Calculates the probability that a first-time buyer will return for a second purchase.
        Also returns the average number of purchases per returning customer.
        Use this for 'how sticky are our customers?' or 'what % of new buyers come back?'
        """
        if not self._has_customer_id:
            return self._missing_col_error("Customer ID")
        if not self._has_invoice:
            return self._missing_col_error("Invoice")

        orders_per_customer = self.df.groupby("Customer ID")["Invoice"].nunique()
        total_customers = len(orders_per_customer)
        returned = int((orders_per_customer > 1).sum())
        avg_orders_returning = (
            float(orders_per_customer[orders_per_customer > 1].mean())
            if returned > 0 else 0.0
        )

        return {
            "total_customers":                  total_customers,
            "returned_for_second_purchase":     returned,
            "repeat_purchase_probability_pct":  round(returned / total_customers * 100, 1) if total_customers > 0 else 0.0,
            "avg_orders_per_returning_customer": round(avg_orders_returning, 1),
        }

    def get_customer_clv_estimate(self, customer_id: int, projection_months: int = 12) -> dict:
        """Estimates the projected Customer Lifetime Value (CLV) for a specific customer.
        Uses their observed purchase frequency and average order value to forecast future value
        over a defined horizon (default: next 12 months).
        Formula: Projected CLV = avg_order_value × (purchases_per_month × projection_months)
        Args:
            customer_id: The numeric customer ID.
            projection_months: How many months ahead to project (default 12, max 36).
        """
        if not self._has_customer_id:
            return self._missing_col_error("Customer ID")
        if not self._has_invoice:
            return self._missing_col_error("Invoice")
        if not self._has_revenue:
            return {"error": "Revenue could not be computed. Ensure both 'Price' and 'Quantity' columns exist in the dataset."}

        try:
            cid_f = float(customer_id)
        except (ValueError, TypeError):
            return {"error": f"Invalid customer ID: '{customer_id}'. Please provide a numeric ID."}

        projection_months = min(projection_months, 36)

        cdf = self.df[self.df["Customer ID"] == cid_f]
        if cdf.empty:
            return {"error": f"Customer ID {customer_id} not found in the dataset."}

        sales = cdf[cdf["Quantity"] > 0] if self._has_quantity else cdf
        if sales.empty:
            return {"error": f"Customer {customer_id} has no recorded purchases."}

        invoices  = sales.groupby("Invoice")["Revenue"].sum()
        aov       = float(invoices.mean())

        if aov == 0:
            return {
                "customer_id":    customer_id,
                "error":          "All recorded orders for this customer have £0 revenue — CLV cannot be projected.",
                "historical_spend_gbp": 0.0,
            }

        total_orders     = len(invoices)
        historical_spend = float(invoices.sum())

        first_purchase   = cdf["InvoiceDate"].min() if self._has_invoice_date else None
        last_purchase    = cdf["InvoiceDate"].max() if self._has_invoice_date else None
        active_days      = (last_purchase - first_purchase).days if (first_purchase is not None and pd.notna(first_purchase) and pd.notna(last_purchase)) else 0
        insufficient_history = total_orders == 1 or active_days == 0
        active_months    = max(active_days / 30.0, 1.0)

        purchases_per_month = total_orders / active_months
        projected_orders    = purchases_per_month * projection_months
        projected_clv       = aov * projected_orders

        note = (
            f"Projected CLV assumes the customer continues buying at their historical rate "
            f"for the next {projection_months} months. Does not account for churn probability."
        )
        if insufficient_history:
            note += (
                " WARNING: This customer has only 1 recorded order or all purchases on the same day. "
                "Purchase frequency cannot be reliably estimated from a single data point — "
                "treat this projection as a rough upper bound, not a reliable forecast."
            )

        return {
            "customer_id":                  customer_id,
            "historical_spend_gbp":         round(historical_spend, 2),
            "total_historical_orders":      total_orders,
            "avg_order_value_gbp":          round(aov, 2),
            "active_months_in_dataset":     round(active_months, 1),
            "purchases_per_month":          round(purchases_per_month, 2),
            "projection_horizon_months":    projection_months,
            "projected_orders":             round(projected_orders, 1),
            "projected_clv_gbp":            round(projected_clv, 2),
            "insufficient_history":         insufficient_history,
            "note":                         note,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utility (shared with other analysts — needed so the agent can resolve names)
    # ──────────────────────────────────────────────────────────────────────────

    def search_products(self, query: str) -> list:
        """Searches the product catalog for items whose description contains the query string.
        ALWAYS call this first when the user refers to a product by a partial or approximate name,
        before calling any other tool that requires an exact product description.
        Returns up to 10 matching product descriptions (exact strings) from the dataset.
        Args:
            query: A partial product name or keyword (case-insensitive, e.g. 'heart candle').
        """
        if not self._has_description:
            return []
        mask = self.df["Description"].str.contains(query, case=False, na=False, regex=False)
        return sorted(self.df[mask]["Description"].unique().tolist())[:10]
