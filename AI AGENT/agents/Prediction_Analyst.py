"""
Prediction_Analyst.py — Pre-built forecasting and predictive analytics tools.

Includes real ML models:
  - Prophet (Facebook/Meta) for time-series revenue forecasting with seasonality + CI
  - Random Forest classifier for churn probability scoring
  - KMeans clustering for RFM-based customer segmentation

Falls back gracefully to pattern-based methods if optional ML libraries are not installed.
"""

import warnings
import pandas as pd
import numpy as np

# Heavy ML libraries (Prophet, sklearn, shap, mlxtend) are imported lazily
# inside each method so that app startup does not load all of them at once.
# Python caches imports in sys.modules, so after the first call they are instant.



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

        # ── ML model cache (lazy-trained once per session) ────────────────────
        self._prophet_model        = None   # trained Prophet model
        self._prophet_forecast_df  = None   # last forecast DataFrame
        self._rf_model             = None   # trained RandomForestClassifier
        self._rf_feature_names     = None   # list of feature column names
        self._rf_scaler            = None   # StandardScaler used during training
        self._rf_training_report   = None   # dict with accuracy / recall metrics
        self._kmeans_model         = None   # trained KMeans
        self._kmeans_rfm_df        = None   # RFM DataFrame with cluster labels
        self._kmeans_metrics       = None   # dict: inertia, silhouette score
        self._optimal_k            = None   # best K from silhouette search
        self._k_silhouette_scores  = None   # dict: k -> silhouette score

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _missing_col_error(self, col: str, as_list: bool = False):
        msg = {"error": f"Required column '{col}' is missing from the dataset."}
        return [msg] if as_list else msg

    def _no_date_error(self, as_list: bool = False):
        msg = {"error": "No valid InvoiceDate values found. All dates are missing or corrupt."}
        return [msg] if as_list else msg

    def _build_rfm_dataframe(self) -> "pd.DataFrame | None":
        """
        Builds an RFM (Recency, Frequency, Monetary) feature DataFrame.
        Shared by the churn classifier and KMeans segmentation.

        Columns: customer_id, recency_days, frequency, monetary,
                 avg_basket_size, purchase_count, return_rate,
                 purchase_span_days, std_basket_size.
        Returns None if required columns are missing or data is insufficient.
        """
        if not (self._has_customer_id and self._has_invoice_date
                and self._has_revenue and self._has_valid_dates):
            return None

        sales_df = (
            self._dated_df[self._dated_df["Quantity"] > 0].copy()
            if self._has_quantity
            else self._dated_df.copy()
        )

        if sales_df.empty:
            return None

        rfm = (
            sales_df.groupby("Customer ID")
            .agg(
                last_purchase=("InvoiceDate", "max"),
                first_purchase=("InvoiceDate", "min"),
                frequency=("Invoice", "nunique"),
                monetary=("Revenue", "sum"),
                purchase_count=("Quantity", "sum"),
            )
            .reset_index()
        )

        rfm["recency_days"]      = (self._reference_date - rfm["last_purchase"]).dt.days
        rfm["purchase_span_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days
        rfm["avg_basket_size"]   = rfm["monetary"] / rfm["frequency"]
        rfm = rfm.drop(columns=["last_purchase", "first_purchase"])
        rfm = rfm.rename(columns={"Customer ID": "customer_id"})
        rfm = rfm.dropna()
        rfm = rfm[rfm["monetary"] > 0]

        # ── Return rate: invoices with net-negative quantity per customer ─────
        if self._has_quantity and self._has_invoice and self._has_customer_id:
            returns_df = self.df[self.df["Quantity"] < 0]
            if not returns_df.empty:
                return_invoices = (
                    returns_df.groupby("Customer ID")["Invoice"]
                    .nunique()
                    .rename("return_invoices")
                )
                rfm = rfm.join(return_invoices, on="customer_id", how="left")
                rfm["return_invoices"] = rfm["return_invoices"].fillna(0)
            else:
                rfm["return_invoices"] = 0
        else:
            rfm["return_invoices"] = 0

        rfm["return_rate"] = rfm["return_invoices"] / rfm["frequency"].replace(0, np.nan)
        rfm["return_rate"] = rfm["return_rate"].fillna(0.0)
        rfm = rfm.drop(columns=["return_invoices"])

        # ── Std dev of order values (spending consistency signal) ─────────────
        basket_std = (
            sales_df.groupby(["Customer ID", "Invoice"])["Revenue"]
            .sum()
            .groupby(level=0)
            .std()
            .fillna(0.0)
            .rename("std_basket_size")
        )
        rfm = rfm.join(basket_std, on="customer_id", how="left")
        rfm["std_basket_size"] = rfm["std_basket_size"].fillna(0.0)

        return rfm

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
            "reference_date":          str(self._reference_date.date()),
            "days_inactive_threshold": days_inactive,
            "total_customers":         total,
            "at_risk_customers":       at_risk,
            "healthy_customers":       healthy,
            "churn_risk_pct":          round(at_risk / total * 100, 1) if total > 0 else 0.0,
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
    # ML Churn Classifier
    # ──────────────────────────────────────────────────────────────────────────

    def get_churn_probability_scores(
        self,
        churn_threshold_days: int = 90,
        top_n: int = 20,
    ) -> dict:
        """Trains a Random Forest churn classifier on RFM features and returns
        ML-derived churn probability scores for customers. More accurate than a
        simple inactivity threshold — the model learns from 8 features:
        Recency, Frequency, Monetary, Basket Size, Purchase Count, Return Rate,
        Purchase Span, and Spending Variability (std basket size).

        Return Rate is derived from negative-Quantity rows (refunds/cancellations)
        and acts as a churn signal — customers who return items frequently are
        statistically more likely to churn.

        Use this for deep churn analysis: 'who is most likely to leave?',
        'what drives churn?', 'show me churn probabilities with feature importance'.

        Returns model accuracy metrics, top at-risk customers ranked by ML
        churn probability, feature importances, and optional SHAP values.

        Args:
            churn_threshold_days: Days inactive used to label churned=1 (default 90).
            top_n: How many high-risk customers to return (default 20, max 50).
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report as _sklearn_cr
        except ImportError:
            return {"error": "scikit-learn is not installed. Run: pip install scikit-learn>=1.3.0"}

        rfm = self._build_rfm_dataframe()
        if rfm is None or len(rfm) < 30:
            return {"error": "Insufficient data to train churn model (need ≥30 customers with complete RFM data)."}

        top_n = min(top_n, 50)

        # ── Label: churned = 1 if inactive ≥ threshold ────────────────────────
        rfm = rfm.copy()
        rfm["churned"] = (rfm["recency_days"] >= churn_threshold_days).astype(int)

        feature_cols = [
            "recency_days", "frequency", "monetary", "avg_basket_size",
            "purchase_count", "return_rate", "purchase_span_days", "std_basket_size",
        ]
        X = rfm[feature_cols].values
        y = rfm["churned"].values

        if len(set(y)) < 2:
            return {
                "error": (
                    f"All customers are in one churn class with a {churn_threshold_days}-day threshold. "
                    "Try adjusting churn_threshold_days."
                )
            }

        # ── Train (cached; retrain only if feature set changes) ──────────────
        if self._rf_model is None or self._rf_feature_names != feature_cols:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            rf.fit(X_train_s, y_train)

            report = _sklearn_cr(y_test, rf.predict(X_test_s), output_dict=True, zero_division=0)

            self._rf_model         = rf
            self._rf_scaler        = scaler
            self._rf_feature_names = feature_cols
            self._rf_training_report = {
                "accuracy":          round(report["accuracy"] * 100, 1),
                "precision_churned": round(report.get("1", {}).get("precision", 0) * 100, 1),
                "recall_churned":    round(report.get("1", {}).get("recall", 0) * 100, 1),
                "f1_churned":        round(report.get("1", {}).get("f1-score", 0) * 100, 1),
                "training_samples":  len(X_train),
            }

        # ── Score all customers ───────────────────────────────────────────────
        X_all_s    = self._rf_scaler.transform(X)
        churn_probs = self._rf_model.predict_proba(X_all_s)[:, 1]
        rfm         = rfm.copy()
        rfm["churn_probability"] = churn_probs

        # ── Feature importances ───────────────────────────────────────────────
        importances = self._rf_model.feature_importances_
        feature_importances = sorted(
            [
                {"feature": f, "importance_pct": round(float(imp) * 100, 1)}
                for f, imp in zip(feature_cols, importances)
            ],
            key=lambda x: x["importance_pct"],
            reverse=True,
        )

        # ── SHAP (optional, TreeExplainer fast path) ──────────────────────────
        shap_summary = None
        try:
            import shap as _shap_lib
            _shap_ok = True
        except ImportError:
            _shap_ok = False
        if _shap_ok:
            try:
                explainer   = _shap_lib.TreeExplainer(self._rf_model)
                sample_X    = X_all_s[: min(200, len(X_all_s))]
                shap_values = explainer.shap_values(sample_X)
                # Binary RF: shap_values is a list [class0_arr, class1_arr]
                class1_shap = shap_values[1] if isinstance(shap_values, list) else shap_values
                mean_abs    = np.abs(class1_shap).mean(axis=0)
                shap_summary = sorted(
                    [
                        {"feature": f, "mean_abs_shap": round(float(v), 4)}
                        for f, v in zip(feature_cols, mean_abs)
                    ],
                    key=lambda x: x["mean_abs_shap"],
                    reverse=True,
                )
            except Exception:
                shap_summary = None  # SHAP failure is non-fatal

        # ── Top at-risk list ──────────────────────────────────────────────────
        top_risk = rfm.sort_values("churn_probability", ascending=False).head(top_n)
        high_risk_list = []
        for _, row in top_risk.iterrows():
            try:
                cid = int(row["customer_id"])
            except (ValueError, TypeError):
                cid = row["customer_id"]
            high_risk_list.append({
                "customer_id":       cid,
                "churn_probability": round(float(row["churn_probability"]) * 100, 1),
                "recency_days":      int(row["recency_days"]),
                "frequency":         int(row["frequency"]),
                "monetary_gbp":      round(float(row["monetary"]), 2),
            })

        return {
            "model_metadata":          self._rf_training_report,
            "churn_threshold_days":    churn_threshold_days,
            "total_customers_scored":  len(rfm),
            "high_risk_count":         int((rfm["churn_probability"] >= 0.5).sum()),
            "high_risk_customers":     high_risk_list,
            "feature_importances":     feature_importances,
            "shap_summary":            shap_summary,
            "returns_signal_note": (
                "return_rate captures refund/cancellation behaviour as a churn signal. "
                "Customers with high return rates are statistically more likely to churn. "
                "purchase_span_days rewards long-tenure customers; std_basket_size "
                "captures spending consistency."
            ),
            "note": (
                "Churn probability is ML-estimated (0–100%). "
                "Customers with score ≥50% are flagged high-risk. "
                "Model: Random Forest trained on 8 features — Recency, Frequency, "
                "Monetary, Basket Size, Purchase Count, Return Rate, "
                "Purchase Span, and Spending Variability."
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # KMeans Customer Segmentation
    # ──────────────────────────────────────────────────────────────────────────

    def _find_optimal_k(self, X_scaled: "np.ndarray", k_range=range(2, 9)) -> int:
        """
        Runs KMeans for each k in k_range, computes silhouette scores, and
        returns the k with the highest score. Caches scores in _k_silhouette_scores.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        scores: dict[int, float] = {}
        for k in k_range:
            if len(X_scaled) < k * 5:
                break
            km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=200)
            labels = km.fit_predict(X_scaled)
            sample_size = min(500, len(X_scaled))
            idx = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
            scores[k] = float(silhouette_score(X_scaled[idx], labels[idx]))
        self._k_silhouette_scores = scores
        return max(scores, key=scores.get) if scores else 4

    def get_customer_segments(self, n_clusters: int = None) -> dict:
        """Performs RFM-based customer segmentation using KMeans clustering.
        Automatically determines the optimal number of clusters (K) via
        Silhouette Analysis across k=2..8, then assigns business-meaningful
        labels based on Recency, Frequency, and Monetary centroid rankings:
        Champions (best), Loyal Customers, At-Risk, Hibernating, Lost, New Customers.

        Use this for 'segment our customers', 'who are our Champions?',
        'which customers are at risk of leaving?', 'RFM analysis', 'clustering'.

        Returns per-segment statistics (size, avg RFM), silhouette quality score,
        the full K-search silhouette table, and a cluster summary.

        Args:
            n_clusters: Number of segments to create. Pass None (default) to
                        auto-detect the optimal K via Silhouette Analysis (range 2–8).
                        Pass an explicit integer to override (clamped to 2–8).
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            return {"error": "scikit-learn is not installed. Run: pip install scikit-learn>=1.3.0"}

        rfm = self._build_rfm_dataframe()
        if rfm is None or len(rfm) < 10:
            return {"error": "Insufficient customer data for segmentation (need ≥10 customers with complete RFM data)."}

        # ── Scale features ────────────────────────────────────────────────────
        features = ["recency_days", "frequency", "monetary"]
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(rfm[features].values)

        # ── Determine K ───────────────────────────────────────────────────────
        if n_clusters is None:
            # Auto-select: run silhouette search only when cache is empty
            if self._optimal_k is None:
                self._optimal_k = self._find_optimal_k(X_scaled)
            n_clusters = self._optimal_k
        else:
            n_clusters = max(2, min(int(n_clusters), 8))

        if len(rfm) < n_clusters * 5:
            return {"error": f"Insufficient customer data for {n_clusters}-cluster segmentation."}

        if self._kmeans_model is None or self._kmeans_model.n_clusters != n_clusters:
            km = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init="auto",
                max_iter=300,
            )
            km.fit(X_scaled)

            sample_size = min(500, len(X_scaled))
            rng         = np.random.RandomState(42)
            idx         = rng.choice(len(X_scaled), sample_size, replace=False)
            sil_score   = float(silhouette_score(X_scaled[idx], km.labels_[idx]))

            self._kmeans_model   = km
            self._kmeans_metrics = {
                "n_clusters":        n_clusters,
                "inertia":           round(float(km.inertia_), 2),
                "silhouette_score":  round(sil_score, 4),
                "k_selected_by":     "silhouette_auto" if self._optimal_k == n_clusters else "user_override",
                "silhouette_scores_by_k": {
                    str(k): round(v, 4)
                    for k, v in (self._k_silhouette_scores or {}).items()
                },
            }

        rfm = rfm.copy()
        rfm["cluster"] = self._kmeans_model.labels_

        # ── Assign human labels by centroid RFM ranking ───────────────────────
        cluster_stats = (
            rfm.groupby("cluster")
            .agg(
                avg_recency=("recency_days", "mean"),
                avg_frequency=("frequency", "mean"),
                avg_monetary=("monetary", "mean"),
                count=("customer_id", "count"),
            )
            .reset_index()
        )

        # Lower recency = better; higher freq + monetary = better
        cluster_stats["rfm_score"] = (
            -cluster_stats["avg_recency"].rank()
            + cluster_stats["avg_frequency"].rank()
            + cluster_stats["avg_monetary"].rank()
        )

        _label_map = {
            0: "Champions",
            1: "Loyal Customers",
            2: "At-Risk",
            3: "Hibernating",
            4: "Lost",
            5: "New Customers",
        }
        sorted_clusters = (
            cluster_stats.sort_values("rfm_score", ascending=False)
            .reset_index(drop=True)
        )
        sorted_clusters["label"] = [
            _label_map.get(i, f"Segment {i + 1}") for i in range(len(sorted_clusters))
        ]

        label_by_cluster = dict(zip(sorted_clusters["cluster"], sorted_clusters["label"]))
        rfm["segment_label"] = rfm["cluster"].map(label_by_cluster)
        self._kmeans_rfm_df  = rfm

        # ── Build output ──────────────────────────────────────────────────────
        cluster_summary = []
        for _, row in sorted_clusters.iterrows():
            cluster_summary.append({
                "label":             row["label"],
                "customer_count":    int(row["count"]),
                "pct_of_total":      round(float(row["count"]) / len(rfm) * 100, 1),
                "avg_recency_days":  round(float(row["avg_recency"]), 1),
                "avg_frequency":     round(float(row["avg_frequency"]), 1),
                "avg_monetary_gbp":  round(float(row["avg_monetary"]), 2),
            })

        segment_counts = {
            str(k): int(v)
            for k, v in rfm["segment_label"].value_counts().items()
        }

        return {
            "cluster_summary":    cluster_summary,
            "model_metadata":     self._kmeans_metrics,
            "all_segments_count": segment_counts,
            "note": (
                "Segments derived from Recency, Frequency, and Monetary value "
                "using KMeans clustering. Optimal K chosen automatically via "
                "Silhouette Analysis — see model_metadata.silhouette_scores_by_k. "
                "Labels ranked by combined RFM score: "
                "Champions have the best profile (recent, frequent, high spend)."
            ),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Revenue forecasting (Prophet + linear fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def get_revenue_forecast(self, horizon_months: int = 3) -> dict:
        """Forecasts monthly revenue for the next N months using Prophet time-series
        modelling with yearly/weekly seasonality, UK holiday effects, and 95%
        confidence intervals (yhat_lower / yhat_upper).

        Use this for 'what will revenue look like next quarter/month?',
        'forecast revenue', 'predict next month sales'.

        Returns predicted revenue with confidence bounds for each future month,
        plus model metadata (training period, seasonality flags).
        Falls back to linear regression if Prophet is not installed.

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

        try:
            from prophet import Prophet
            _prophet_ok = True
        except ImportError:
            _prophet_ok = False

        if not _prophet_ok:
            return self._get_revenue_forecast_linear(horizon_months)

        # ── Build daily revenue series ────────────────────────────────────────
        sales_df = (
            self._dated_df[self._dated_df["Quantity"] > 0].copy()
            if self._has_quantity
            else self._dated_df.copy()
        )

        daily = (
            sales_df.set_index("InvoiceDate")
            .resample("D")["Revenue"]
            .sum()
            .reset_index()
        )
        daily.columns = ["ds", "y"]
        daily = daily[daily["y"] > 0]

        if len(daily) < 30:
            return self._get_revenue_forecast_linear(horizon_months)

        # ── Train Prophet (cached per session) ────────────────────────────────
        if self._prophet_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = Prophet(  # noqa: F821 — imported above
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                    interval_width=0.95,
                    changepoint_prior_scale=0.15,   # was 0.05 — captures real trend inflections
                    seasonality_prior_scale=15.0,   # stronger e-commerce seasonality signal
                )
                m.add_country_holidays(country_name="UK")
                m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
                m.fit(daily)
            self._prophet_model = m

        # ── Generate forecast (exact daily range → trimmed to N months) ───────
        last_date    = daily["ds"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon_months * 31,   # over-generate; trimmed to horizon_months below
            freq="D",
        )
        future = pd.DataFrame({
            "ds": pd.concat(
                [daily["ds"], pd.Series(future_dates)], ignore_index=True
            )
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_df = self._prophet_model.predict(future)

        self._prophet_forecast_df = forecast_df

        # ── Aggregate daily → monthly ─────────────────────────────────────────
        forecast_df = forecast_df.copy()
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])
        last_train_date   = daily["ds"].max()
        future_only       = forecast_df[forecast_df["ds"] > last_train_date].copy()
        future_only["month"] = future_only["ds"].dt.to_period("M")

        monthly_forecast = (
            future_only.groupby("month")
            .agg(yhat=("yhat", "sum"), yhat_lower=("yhat_lower", "sum"), yhat_upper=("yhat_upper", "sum"))
            .reset_index()
            .head(horizon_months)
        )

        # Last 3 historical months (fitted values for context)
        hist_only = forecast_df[forecast_df["ds"] <= last_train_date].copy()
        hist_only["month"] = hist_only["ds"].dt.to_period("M")
        historical_monthly = (
            hist_only.groupby("month")["yhat"].sum().tail(3)
        )

        forecasts: dict[str, dict] = {}
        for _, row in monthly_forecast.iterrows():
            forecasts[str(row["month"])] = {
                "predicted_gbp":   round(float(max(row["yhat"], 0.0)), 2),
                "lower_bound_gbp": round(float(max(row["yhat_lower"], 0.0)), 2),
                "upper_bound_gbp": round(float(max(row["yhat_upper"], 0.0)), 2),
            }

        if len(monthly_forecast) >= 2:
            slope = float(monthly_forecast["yhat"].iloc[-1]) - float(monthly_forecast["yhat"].iloc[0])
            trend = "upward" if slope > 500 else "downward" if slope < -500 else "flat"
        else:
            trend = "insufficient data"

        return {
            "forecast":               forecasts,
            "trend":                  trend,
            "model":                  "Prophet (multiplicative, changepoint_prior=0.15, monthly+weekly+yearly seasonality, UK holidays, 95% CI)",
            "training_days":          len(daily),
            "training_start":         str(daily["ds"].min().date()),
            "training_end":           str(daily["ds"].max().date()),
            "horizon_months":         horizon_months,
            "last_3_months_yhat_gbp": {
                str(k): round(float(v), 2) for k, v in historical_monthly.items()
            },
            "warning": (
                "Forecast based on historical patterns with seasonality modelling. "
                "Cannot predict external shocks, promotions, or supply disruptions."
            ),
        }

    def _get_revenue_forecast_linear(self, horizon_months: int = 3) -> dict:
        """Fallback linear regression forecast (used when Prophet is not installed)."""
        horizon_months = min(horizon_months, 12)

        sales_df = (
            self._dated_df[self._dated_df["Quantity"] > 0].copy()
            if self._has_quantity
            else self._dated_df.copy()
        )
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

        median_rev = float(monthly["revenue"].median())
        outlier_warning = None
        if median_rev > 0:
            spike_mask = monthly["revenue"] > 5 * median_rev
            if spike_mask.any():
                worst = float(monthly.loc[spike_mask, "revenue"].max())
                outlier_warning = (
                    f"WARNING: {spike_mask.sum()} month(s) contain revenue spikes "
                    f"({worst:,.0f} £ vs median {median_rev:,.0f} £) >5× the typical "
                    "month and will heavily skew this linear forecast."
                )

        x = np.arange(len(monthly))
        y = monthly["revenue"].values
        slope, intercept = np.polyfit(x, y, 1)

        trend = "upward" if slope > 0 else "downward" if slope < 0 else "flat"

        last_month = monthly["month"].iloc[-1]
        forecasts: dict[str, float] = {}
        for i in range(1, horizon_months + 1):
            future_x     = len(monthly) - 1 + i
            predicted    = intercept + slope * future_x
            future_month = (last_month + pd.DateOffset(months=i)).to_period("M")
            forecasts[str(future_month)] = round(float(max(predicted, 0.0)), 2)

        result: dict = {
            "forecast":                   forecasts,
            "trend":                      trend,
            "model":                      "linear regression (Prophet not installed)",
            "monthly_slope_gbp":          round(float(slope), 2),
            "last_actual_month":          str(monthly["month"].iloc[-1].to_period("M")),
            "last_actual_revenue_gbp":    round(float(monthly["revenue"].iloc[-1]), 2),
            "historical_avg_monthly_gbp": round(float(monthly["revenue"].mean()), 2),
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
            mask.reindex(self._dated_df.index, fill_value=False)
            & qty_filter.reindex(self._dated_df.index, fill_value=False)
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
            "product":             product_desc,
            "trend":               trend,
            "monthly_slope_units": round(float(slope), 2),
            "total_units_sold":    int(product_df["Quantity"].sum()),
            "months_of_data":      len(monthly_qty),
            "monthly_breakdown":   monthly_dict,
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
    # Market Basket Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def get_market_basket_rules(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.3,
        top_n: int = 15,
    ) -> dict:
        """Discovers which products are frequently bought together using
        Association Rule Mining (FP-Growth algorithm when mlxtend is installed,
        co-occurrence matrix fallback otherwise).

        Use this for: 'frequently bought together', 'cross-sell opportunities',
        'product pairs', 'what do customers buy with X?', 'market basket analysis',
        'association rules', 'bundling recommendations'.

        Returns the top N rules sorted by lift (descending). Lift > 1 means
        items are bought together more than by chance; Lift > 3 is a strong signal.

        Args:
            min_support: Minimum fraction of transactions containing the itemset
                         (default 0.01 = 1%). Lower values find rarer pairs.
            min_confidence: Minimum P(consequent | antecedent) (default 0.3 = 30%).
            top_n: How many rules to return (default 15, max 50).
        """
        if not self._has_invoice:
            return self._missing_col_error("Invoice")
        if not self._has_description:
            return self._missing_col_error("Description")
        if not self._has_quantity:
            return self._missing_col_error("Quantity")

        top_n = min(top_n, 50)

        # ── Filter to positive sales only ─────────────────────────────────────
        sales_df = self._dated_df[self._dated_df["Quantity"] > 0].copy() if self._has_valid_dates \
                   else self.df[self.df["Quantity"] > 0].copy()
        sales_df = sales_df.dropna(subset=["Invoice", "Description"])

        if sales_df.empty:
            return {"error": "No valid sales transactions found for basket analysis."}

        # ── Performance guard: sample large datasets ──────────────────────────
        MAX_INVOICES = 150_000
        unique_invoices = sales_df["Invoice"].nunique()
        sampled = False

        if unique_invoices > MAX_INVOICES:
            rng = np.random.RandomState(42)
            sampled_inv = rng.choice(
                sales_df["Invoice"].unique(), MAX_INVOICES, replace=False
            )
            sales_df = sales_df[sales_df["Invoice"].isin(sampled_inv)]
            sampled = True
            unique_invoices = MAX_INVOICES

        # ── Build basket: Invoice × Product boolean matrix ────────────────────
        try:
            from mlxtend.frequent_patterns import fpgrowth as _fpgrowth
            from mlxtend.frequent_patterns import association_rules as _assoc_rules
            from mlxtend.preprocessing import TransactionEncoder as _TransactionEncoder
            _mlxtend_ok = True
        except ImportError:
            _mlxtend_ok = False

        if _mlxtend_ok:
            # FP-Growth path (fast, memory-efficient)
            basket_sets = (
                sales_df.groupby("Invoice")["Description"]
                .apply(list)
                .tolist()
            )

            te = _TransactionEncoder()
            te_array = te.fit_transform(basket_sets)
            basket_df = pd.DataFrame(te_array, columns=te.columns_)

            frequent_items = _fpgrowth(
                basket_df,
                min_support=min_support,
                use_colnames=True,
            )

            if frequent_items.empty:
                return {
                    "rules": [],
                    "total_rules_found": 0,
                    "total_transactions_analysed": unique_invoices,
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "engine": "FP-Growth (mlxtend)",
                    "sampled": sampled,
                    "note": (
                        f"No itemsets met the min_support={min_support} threshold. "
                        "Try lowering min_support (e.g. 0.005)."
                    ),
                }

            rules_df = _assoc_rules(
                frequent_items,
                metric="confidence",
                min_threshold=min_confidence,
            )

            if rules_df.empty:
                return {
                    "rules": [],
                    "total_rules_found": 0,
                    "total_transactions_analysed": unique_invoices,
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "engine": "FP-Growth (mlxtend)",
                    "sampled": sampled,
                    "note": (
                        f"No rules met min_confidence={min_confidence}. "
                        "Try lowering min_confidence (e.g. 0.2)."
                    ),
                }

            rules_df = rules_df.sort_values("lift", ascending=False).head(top_n)

            rules_out = [
                {
                    "antecedent":  ", ".join(sorted(row["antecedents"])),
                    "consequent":  ", ".join(sorted(row["consequents"])),
                    "support":     round(float(row["support"]), 4),
                    "confidence":  round(float(row["confidence"]), 3),
                    "lift":        round(float(row["lift"]), 2),
                }
                for _, row in rules_df.iterrows()
            ]
            engine = "FP-Growth (mlxtend)"

        else:
            # ── Fallback: co-occurrence matrix ────────────────────────────────
            import itertools
            from collections import defaultdict

            item_counts: dict = defaultdict(int)
            pair_counts: dict = defaultdict(int)
            n_transactions = 0

            for _, grp in sales_df.groupby("Invoice")["Description"]:
                items = list(grp.unique())
                if len(items) < 2:
                    continue
                n_transactions += 1
                for item in items:
                    item_counts[item] += 1
                for a, b in itertools.combinations(sorted(items), 2):
                    pair_counts[(a, b)] += 1

            if n_transactions == 0 or not pair_counts:
                return {
                    "rules": [],
                    "total_rules_found": 0,
                    "total_transactions_analysed": unique_invoices,
                    "min_support": min_support,
                    "min_confidence": min_confidence,
                    "engine": "co-occurrence matrix (mlxtend not installed)",
                    "sampled": sampled,
                    "note": "No multi-item baskets found.",
                }

            rules_out = []
            for (a, b), co_count in pair_counts.items():
                support = co_count / n_transactions
                if support < min_support:
                    continue
                conf_ab = co_count / item_counts[a]
                conf_ba = co_count / item_counts[b]
                lift = support / (
                    (item_counts[a] / n_transactions) * (item_counts[b] / n_transactions)
                )
                if conf_ab >= min_confidence:
                    rules_out.append({
                        "antecedent": a, "consequent": b,
                        "support": round(support, 4),
                        "confidence": round(conf_ab, 3),
                        "lift": round(lift, 2),
                    })
                if conf_ba >= min_confidence:
                    rules_out.append({
                        "antecedent": b, "consequent": a,
                        "support": round(support, 4),
                        "confidence": round(conf_ba, 3),
                        "lift": round(lift, 2),
                    })

            rules_out.sort(key=lambda x: x["lift"], reverse=True)
            rules_out = rules_out[:top_n]
            engine = "co-occurrence matrix (mlxtend not available)"

        return {
            "rules":                        rules_out,
            "total_rules_found":            len(rules_out),
            "total_transactions_analysed":  unique_invoices,
            "min_support":                  min_support,
            "min_confidence":               min_confidence,
            "engine":                       engine,
            "sampled":                      sampled,
            "note": (
                "Lift > 1 means items are bought together more than by chance. "
                "Lift > 3 is a strong cross-sell signal. "
                "Confidence = P(consequent | antecedent purchased)."
            ),
        }

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
            "total_customers":                   total_customers,
            "returned_for_second_purchase":      returned,
            "repeat_purchase_probability_pct":   round(returned / total_customers * 100, 1) if total_customers > 0 else 0.0,
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

        invoices         = sales.groupby("Invoice")["Revenue"].sum()
        aov              = float(invoices.mean())

        if aov == 0:
            return {
                "customer_id":          customer_id,
                "error":                "All recorded orders for this customer have £0 revenue — CLV cannot be projected.",
                "historical_spend_gbp": 0.0,
            }

        total_orders     = len(invoices)
        historical_spend = float(invoices.sum())

        first_purchase  = cdf["InvoiceDate"].min() if self._has_invoice_date else None
        last_purchase   = cdf["InvoiceDate"].max() if self._has_invoice_date else None
        active_days     = (
            (last_purchase - first_purchase).days
            if (first_purchase is not None and pd.notna(first_purchase) and pd.notna(last_purchase))
            else 0
        )
        insufficient_history = total_orders == 1 or active_days == 0
        active_months        = max(active_days / 30.0, 1.0)

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
                "Purchase frequency cannot be reliably estimated — treat as a rough upper bound."
            )

        return {
            "customer_id":               customer_id,
            "historical_spend_gbp":      round(historical_spend, 2),
            "total_historical_orders":   total_orders,
            "avg_order_value_gbp":       round(aov, 2),
            "active_months_in_dataset":  round(active_months, 1),
            "purchases_per_month":       round(purchases_per_month, 2),
            "projection_horizon_months": projection_months,
            "projected_orders":          round(projected_orders, 1),
            "projected_clv_gbp":         round(projected_clv, 2),
            "insufficient_history":      insufficient_history,
            "note":                      note,
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
