"""Sales Analytics Dashboard - Streamlit app."""
import pandas as pd
import plotly.express as px
import streamlit as st

import utils

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def run_pipeline(raw: pd.DataFrame, drop_outliers: bool):
    return utils.preprocess(raw, drop_outliers)


def render_validation(raw: pd.DataFrame) -> None:
    info = utils.validate_data(raw)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", info["rows"])
    c2.metric("Columns", info["columns"])
    c3.metric("Duplicate records", info["duplicate_count"])
    if info["missing_required"]:
        st.error(f"Missing required columns: {', '.join(info['missing_required'])}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Column names & data types**")
        st.dataframe(info["dtypes"], width="stretch", hide_index=True)
    with right:
        st.markdown("**Missing values**")
        st.dataframe(info["missing"], width="stretch", hide_index=True)
    if info["duplicate_count"]:
        with st.expander("View duplicate records"):
            st.dataframe(info["duplicates"], width="stretch")


def render_kpis(df: pd.DataFrame) -> None:
    k = utils.calculate_kpis(df)
    cols = st.columns(5)
    cols[0].metric("Total Sales", f"${k['Total Sales']:,.0f}")
    cols[1].metric("Total Profit", f"${k['Total Profit']:,.0f}")
    cols[2].metric("Total Orders", f"{k['Total Orders']:,}")
    cols[3].metric("Total Quantity Sold", f"{k['Total Quantity']:,}")
    cols[4].metric("Profit Margin", f"{k['Profit Margin (%)']:.2f}%")


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    dmin, dmax = df["Order Date"].min().date(), df["Order Date"].max().date()
    date_range = st.sidebar.date_input("Order date range", (dmin, dmax),
                                       min_value=dmin, max_value=dmax)
    if len(date_range) != 2:
        st.sidebar.info("Select both a start and an end date.")
        date_range = (dmin, dmax)

    def multi(label, col):
        options = sorted(df[col].dropna().unique())
        return st.sidebar.multiselect(label, options, default=options)

    return utils.apply_filters(
        df, date_range,
        regions=multi("Region", "Region"),
        categories=multi("Category", "Category"),
        segments=multi("Segment", "Segment"),
        ship_modes=multi("Ship Mode", "Ship Mode"),
    )


def render_charts(df: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    monthly = df.groupby("Month", as_index=False)[["Sales", "Profit"]].sum()
    c1.plotly_chart(px.line(monthly, x="Month", y=["Sales", "Profit"], markers=True,
                            title="Monthly Sales & Profit Trend"), width="stretch")
    cat = df.groupby("Category", as_index=False)[["Sales", "Profit"]].sum()
    c2.plotly_chart(px.bar(cat, x="Category", y=["Sales", "Profit"], barmode="group",
                           title="Sales & Profit by Category"), width="stretch")

    c3, c4 = st.columns(2)
    reg = df.groupby("Region", as_index=False)["Sales"].sum()
    c3.plotly_chart(px.pie(reg, names="Region", values="Sales", hole=0.45,
                           title="Sales Share by Region"), width="stretch")
    c4.plotly_chart(px.scatter(df, x="Sales", y="Profit", color="Category", size="Quantity",
                               hover_data=["Order ID", "Product Name"],
                               title="Sales vs Profit (bubble = quantity)"),
                    width="stretch")

    c5, c6 = st.columns(2)
    top = (df.groupby("Sub-Category", as_index=False)["Sales"].sum()
             .nlargest(10, "Sales").sort_values("Sales"))
    c5.plotly_chart(px.bar(top, x="Sales", y="Sub-Category", orientation="h",
                           title="Top Sub-Categories by Sales"), width="stretch")
    c6.plotly_chart(px.box(df, x="Segment", y="Profit", color="Segment",
                           title="Profit Distribution by Segment"), width="stretch")


# --------------------------------------------------------------------- main
def main() -> None:
    st.title("Sales Analytics Dashboard")
    st.caption("Upload a CSV, Excel or JSON file to validate, clean and explore it.")

    uploaded = st.sidebar.file_uploader("Upload data", type=["csv", "xlsx", "json"])
    drop_outliers = st.sidebar.checkbox("Remove extreme outliers (3×IQR)", value=True)

    if uploaded is None:
        st.info("👈 Upload a file from the sidebar to begin.")
        return

    try:
        raw = utils.load_file(uploaded)
    except Exception as exc:  # unreadable / malformed file
        st.error(f"Could not read the file: {exc}")
        return
    if raw.empty:
        st.warning("The uploaded file contains no rows.")
        return

    missing = [c for c in utils.REQUIRED_COLS if c not in raw.columns]
    tab_val, tab_clean, tab_dash = st.tabs(["1. Validation", "2. Cleaning & Stats", "3. Dashboard"])

    with tab_val:
        st.subheader("Raw data validation")
        render_validation(raw)
        with st.expander("Preview raw data"):
            st.dataframe(raw.head(50), width="stretch")

    if missing:
        st.error(f"Cannot continue - required columns missing: {', '.join(missing)}")
        return

    try:
        clean, log = run_pipeline(raw, drop_outliers)
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")
        return
    if clean.empty:
        st.warning("No valid rows remain after cleaning.")
        return

    with tab_clean:
        st.subheader("Preprocessing summary")
        st.json(log)
        st.subheader("Statistical summary")
        st.dataframe(utils.numeric_summary(clean), width="stretch")
        group_col = st.selectbox("Summarise by", ["Category", "Sub-Category", "Region",
                                                  "Segment", "Ship Mode", "State", "Year"])
        st.dataframe(utils.categorical_summary(clean, group_col), width="stretch")
        st.download_button("Download cleaned CSV", clean.to_csv(index=False).encode(),
                           "cleaned_data.csv", "text/csv")

    with tab_dash:
        filtered = sidebar_filters(clean)
        if filtered.empty:
            st.warning("No data matches the selected filters.")
            return
        render_kpis(filtered)
        st.divider()
        render_charts(filtered)


if __name__ == "__main__":
    main()
