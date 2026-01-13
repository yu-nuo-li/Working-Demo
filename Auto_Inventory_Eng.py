# =========================
# This is an automated inventory managment system.
# =========================
# Required Sheets (with required columns）
# - Inventory Sheet: must include [Name (Reference), Date, SKU, Opening Stock (Carryover), Daily Inbound, Ending Stock]
# - Sales Sheet: must include [SKU, Quantity], optional [Date]
# - Exchange Sheet: must include [Original SKU, New SKU], optional [Quantity, Date]
# - Workflow:
#   1. Yesterday Ending Stock → Today Opening Stock (Carryover)
#   2. Read sales / exchanges
#   3. Calculate ending stock
# - Daily sales are used only as temporary variables for calculation & display,
#   and will NOT be written back to the inventory table


import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
import os
import re

st.set_page_config(page_title="Inventory Management System", layout="centered")
st.title("Inventory Management System")

# ========== File Upload ==========
stock_file = st.file_uploader(
    "Upload [Inventory CSV] (Required | long format: Name (Reference) / Date / SKU / Opening Stock (Carryover) / Daily Inbound / Ending Stock)",
    type=["csv"]
)

sales_files = st.file_uploader(
    "Upload [Sales CSV] (currently supports one file)",
    type=["csv"],
    accept_multiple_files=True
)

if "show_exchange" not in st.session_state:
    st.session_state.show_exchange = False

if st.button("Any influencer exchange orders? (Click to toggle)"):
    st.session_state.show_exchange = not st.session_state.show_exchange

exchange_df = None
if st.session_state.show_exchange:
    st.info("Please upload the exchange CSV. Required columns: Original SKU / New SKU / Date / Quantity.")
    exchange_file = st.file_uploader("Upload Exchange CSV (Optional)", type=["csv"])
    if exchange_file:
        try:
            exchange_df = pd.read_csv(exchange_file)
            st.success("Exchange file uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to read exchange file: {e}")

st.divider()

# ========== Read & Standardize Inventory Sheet ==========
if stock_file is None:
    st.info("Please upload the inventory CSV first.")
    st.stop()

stock_df = pd.read_csv(stock_file)
stock_df.columns = [c.strip() for c in stock_df.columns]

# Required column check
required_columns = {
    "product_name",
    "date",
    "sku",
    "opening_stock",
    "inbound_qty",
    "ending_stock",
    "safety_stock"
}

missing_cols = required_columns - set(stock_df.columns)
if missing_cols:
    st.error(f"Inventory CSV is missing required columns: {missing_cols}")
    st.stop()

# Standardlize date & SKU
stock_df["date"] = (
    pd.to_datetime(stock_df["date"], errors="coerce")
    .dt.strftime("%Y-%m-%d")
)

stock_df["sku"] = (
    stock_df["sku"]
    .astype(str)
    .str.strip()
    .str.upper()
)

available_dates = sorted(
    d for d in stock_df["date"].dropna().unique()
)

default_date = (
    available_dates[-1]
    if available_dates
    else datetime.today().strftime("%Y-%m-%d")
)

work_date = st.date_input("Select working date for this update", value=pd.to_datetime(default_date)).strftime("%Y-%m-%d")

# =======================
# Start Processing
# =======================
if st.button("Start Processing"):

    if not sales_files:
        st.error("Please upload at least one sales CSV.")
        st.stop()
        
   # ① Auto carryover: yesterday ending → today opening 
    prev_date = (pd.to_datetime(work_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    prev_stock_map = (stock_df.loc[stock_df["date"] == prev_date].drop_duplicates("sku", keep="last").set_index("sku")["ending_stock"].pipe(lambda s: pd.to_numeric(s, errors="coerce"))
        .to_dict())

    today_mask = stock_df["date"] == work_date

    if not today_mask.any():
        base = stock_df.drop_duplicates("sku")

        new_today = pd.DataFrame({
            "sku": base["sku"],
            "date": work_date,
            "opening_stock": base["sku"].map(prev_stock_map),
            "inbound_qty": 0,
            "ending_stock": pd.NA,
            "product_name": base.set_index("sku")["product_name"]
        })

        if "base_opening_stock" in base.columns:
            seed_map = base.set_index("sku")["base_opening_stock"]
            new_today["opening_stock"] = (new_today["opening_stock"].fillna(new_today["sku"].map(seed_map)))

        new_today["opening_stock"] = (pd.to_numeric(new_today["opening_stock"], errors="coerce").fillna(0).astype(int))

        if "safety_stock" in base.columns:
            new_today["safety_stock"] = (new_today["sku"].map(base.set_index("sku")["safety_stock"]))

        keep_cols = [
            c for c in stock_df.columns
            if c in new_today.columns
        ]

        stock_df = pd.concat(
            [stock_df, new_today[keep_cols]],
            ignore_index=True
        )
        today_mask = stock_df["date"] == work_date

    else:
        idx_na = stock_df.index[
            today_mask & stock_df["opening_stock"].isna()
        ]
        stock_df.loc[idx_na, "opening_stock"] = (
            stock_df.loc[idx_na, "sku"].map(prev_stock_map)
        )

        if "base_opening_stock" in stock_df.columns:
            seed_map = (stock_df.drop_duplicates("sku").set_index("sku")["base_opening_stock"].to_dict())
            still_na = stock_df.index[today_mask & stock_df["opening_stock"].isna()]
            stock_df.loc[still_na, "opening_stock"] = (stock_df.loc[still_na, "sku"].map(seed_map))

        stock_df.loc[today_mask, "opening_stock"] = (pd.to_numeric(stock_df.loc[today_mask, "opening_stock"], errors="coerce").fillna(0).astype(int))

    # ② Read & aggregate daily sales
    def read_sales_file(file_obj):
        df = pd.read_csv(file_obj)
        df.columns = [c.strip().lower() for c in df.columns]

        sku_col = next(
            (c for c in df.columns if c in ["sku", "sku_code", "sku_id"]),
            None
        )
        qty_col = next(
            (c for c in df.columns if c in ["qty", "quantity", "sales_qty"]),
            None
        )
        date_col = next(
            (c for c in df.columns if c in ["date", "order_date"]),
            None
        )

        if sku_col is None or qty_col is None:
            st.warning(
                f"Skipped {getattr(file_obj, 'name', 'Unnamed')} "
                "(SKU or quantity column not found)"
            )
            return None

        tmp = df[[sku_col, qty_col] + ([date_col] if date_col else [])].copy()
        tmp.columns = ["sku", "qty"] + (["date"] if date_col else [])

        tmp["sku"] = tmp["sku"].astype(str).str.strip().str.upper()
        tmp["qty"] = pd.to_numeric(tmp["qty"], errors="coerce").fillna(0).astype(int)

        if date_col:
            tmp["date"] = (
                pd.to_datetime(tmp["date"], errors="coerce")
                .dt.strftime("%Y-%m-%d")
            )
            tmp = tmp[tmp["date"] == work_date]

        tmp = tmp[(tmp["sku"] != "") & (tmp["qty"] > 0)]
        return tmp[["sku", "qty"]]
    
    sales_list = [
        x for x in
        (read_sales_file(f) for f in sales_files)
        if x is not None
    ]

    if not sales_list:
        st.error("No valid sales data found for the selected date.")
        st.stop()

    sales_all = (
        pd.concat(sales_list, ignore_index=True)
        .groupby("sku", as_index=False)["qty"]
        .sum()
    )

    # ③ Optional: exchange & return adjustment 
    if st.session_state.show_exchange and exchange_df is not None:
        ex = exchange_df.copy()
        ex.columns = [c.strip().lower() for c in ex.columns]

        def find_col(candidates):
            return next((c for c in ex.columns if c in candidates), None)

        orig_col = find_col({"original_sku", "origsku", "orig_sku"})
        new_col  = find_col({"new_sku", "newsku"})
        qty_col  = find_col({"qty", "quantity"})
        date_col = find_col({"date", "transaction_date", "work_date"})

        if not orig_col or not new_col:
            st.warning("Exchange adjustment skipped: required SKU columns not found.")
        else:
            if qty_col is None:
                ex["qty"] = 1
                qty_col = "qty"

            if date_col:
                ex["date"] = (
                    pd.to_datetime(ex[date_col], errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                )
                ex = ex[ex["date"] == work_date]

            ex[orig_col] = ex[orig_col].astype(str).str.strip().str.upper()
            ex[new_col]  = ex[new_col].astype(str).str.strip().str.upper()
            ex[qty_col]  = pd.to_numeric(ex[qty_col], errors="coerce").fillna(1).astype(int)

            plus_df = (
                ex.groupby(new_col, as_index=False)[qty_col]
                .sum()
                .rename(columns={new_col: "sku", qty_col: "plus_qty"})
            )

            minus_df = (
                ex.groupby(orig_col, as_index=False)[qty_col]
                .sum()
                .rename(columns={orig_col: "sku", qty_col: "minus_qty"})
            )

            adj = (
                sales_all
                .merge(plus_df, how="outer", on="sku")
                .merge(minus_df, how="outer", on="sku")
            )

            for col in ["qty", "plus_qty", "minus_qty"]:
                if col not in adj.columns:
                    adj[col] = 0

            adj[["qty", "plus_qty", "minus_qty"]] = (
                adj[["qty", "plus_qty", "minus_qty"]]
                .fillna(0)
                .astype(int)
            )

            adj["qty"] = adj["qty"] + adj["plus_qty"] - adj["minus_qty"]
            sales_all = adj[["sku", "qty"]]

            st.info("Daily sales have been adjusted based on exchange records.")

    # ④ Calculate ending stock
            day_mask = stock_df["date"] == work_date
            sku_to_qty = dict(zip(sales_all["sku"], sales_all["qty"]))

            sold_today = (stock_df.loc[day_mask, "sku"].map(sku_to_qty).fillna(0).astype(int))

            opening_val = pd.to_numeric(stock_df.loc[day_mask, "opening_stock"], errors="coerce").fillna(0)

            inbound_val = pd.to_numeric(stock_df.loc[day_mask, "inbound_qty"], errors="coerce").fillna(0)

            stock_df.loc[day_mask, "ending_stock"] = (opening_val + inbound_val - sold_today).clip(lower=0).astype(int)

    # ⑤ Daily view & summary
    base_cols = ["product_name", "sku", "opening_stock", "inbound_qty", "ending_stock"]

    today_view = stock_df.loc[day_mask, base_cols].copy()
    today_view.insert(3, "sold_qty", sold_today.values)
    today_view = today_view.sort_values("sku").reset_index(drop=True)
    today_view.index += 1

    total_row = pd.DataFrame([[
        "—",
        "—",
        int(today_view["opening_stock"].sum()),
        int(today_view["inbound_qty"].sum()),
        int(today_view["sold_qty"].sum()),
        int(today_view["ending_stock"].sum())
    ]], columns=["product_name", "sku", "opening_stock", "inbound_qty", "sold_qty", "ending_stock"])

    summary_df = pd.concat([today_view, total_row], ignore_index=True)


    # UI Metrics & Status
    st.subheader(f"Inventory Update Results ({work_date})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Opening Stock", int(total_row["opening_stock"].iloc[0]))
    c2.metric("Total Daily Sales", int(total_row["sold_qty"].iloc[0]))
    c3.metric("Total Ending Stock", int(total_row["ending_stock"].iloc[0]))

    low_stock_count = 0
    if "safety_stock" in stock_df.columns:
        safe = stock_df.loc[day_mask, "safety_stock"]
        endv = stock_df.loc[day_mask, "ending_stock"]
        low_stock_count = int((endv < safe).sum())
    sku_count = today_view.shape[0]

    st.write(
        f"Total SKUs: **{sku_count}**  "
        + (
            f"| Low Stock SKUs: **{low_stock_count}**"
            if low_stock_count > 0
            else "| **Inventory Status: Healthy**"
        )
    )
    # Conditional Styling （low stock/no sales）
    def color_rules(df: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df.index, columns=df.columns)

        if "safety_stock" in stock_df.columns:
            safemap = (stock_df.loc[stock_df["date"] == work_date].drop_duplicates("sku").set_index("sku")["safety_stock"])

            safe_val = df["sku"].map(safemap)
            low_mask = df["ending_stock"] < safe_val.fillna(float("inf"))
            styles.loc[low_mask, ["ending_stock"]] = ("background-color:#FFF7E6; color:#7C2D12;")

        styles.loc[df["sold_qty"] == 0, ["sold_qty"]] = "color:#94A3B8;"
        styles.loc[df["inbound_qty"] > 0, ["inbound_qty"]] = ("background-color:#ECFDF5; color:#065F46;" )

        styles.loc[df["sku"] == "—", :] = ( "font-weight:700; background:#F8FAFC;")

        return styles

    numeric_cols = ["opening_stock","inbound_qty","sold_qty","ending_stock"]

    styled = (summary_df.style.format("{:,.0f}", subset=numeric_cols).set_properties(**{"white-space": "nowrap"}).apply(color_rules, axis=None))

   # Display & Download
    st.dataframe(styled, use_container_width=True)

    st.subheader("Copy Ending Stock (One-Click)")
    st.code(
        "\n".join(
            today_view.loc[today_view["sku"] != "—", "ending_stock"]
            .astype(str)
            .tolist()
        ),
        language="text"
    )

    csv_out = summary_df.to_csv(index_label="index").encode("utf-8-sig")
    st.download_button(
        "Download Inventory Update CSV",
        csv_out,
        file_name=f"inventory_update_{work_date}.csv"
    )

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index_label="index")

    st.download_button(
        "Download Inventory Update Excel",
        out.getvalue(),
        file_name="inventory_update.xlsx"
    )

    # Upload History
    history_file = "upload_history.csv"

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "work_date": work_date,
        "inventory_file": stock_file.name if stock_file else "",
        "sales_file_count": len(sales_files),
        "exchange_enabled": st.session_state.show_exchange
    }

    try:
        if os.path.exists(history_file):
            hist = pd.read_csv(history_file)
            hist = pd.concat([hist, pd.DataFrame([record])], ignore_index=True)
        else:
            hist = pd.DataFrame([record])

        hist.to_csv(history_file, index=False, encoding="utf-8-sig")
        st.success("Upload history recorded successfully.")
    except OSError as e:
        st.warning(f"Failed to save upload history (read-only environment): {e}")