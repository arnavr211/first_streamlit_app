"""
DoorDash New Verticals Analytics - Exploratory Data Analysis
Analyzes one month of delivery data across Grocery, Convenience, Alcohol, and DashMart verticals.
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = '2025 New Verticals Analytics Exercise.xlsx'

def load_data():
    df = pd.read_excel(DATA_FILE, sheet_name='Dataset', header=1)
    return df

def delivery_level_summary(df):
    deliv = df.groupby('DELIVERY_UUID').first()
    print("=" * 60)
    print("DELIVERY-LEVEL OVERVIEW")
    print("=" * 60)
    print(f"Total unique deliveries: {len(deliv):,}")
    print(f"Date range: {deliv['DELIV_CREATED_AT'].min()} to {deliv['DELIV_CREATED_AT'].max()}")
    print(f"Cancelled deliveries: {deliv['DELIV_CANCELLED_AT'].notna().sum()} ({deliv['DELIV_CANCELLED_AT'].notna().mean()*100:.1f}%)")
    print(f"Deliveries 20+ min late: {deliv['DELIV_IS_20_MIN_LATE'].sum()} ({deliv['DELIV_IS_20_MIN_LATE'].mean()*100:.1f}%)")
    print(f"Missing/incorrect reports: {deliv['DELIV_MISSING_INCORRECT_REPORT'].sum()} ({deliv['DELIV_MISSING_INCORRECT_REPORT'].mean()*100:.1f}%)")
    return deliv

def store_breakdown(deliv):
    print("\n" + "=" * 60)
    print("STORE BREAKDOWN")
    print("=" * 60)
    store_deliv = deliv.groupby('DELIV_STORE_NAME').agg(
        num_deliveries=('DELIV_CREATED_AT', 'count'),
        pct_late=('DELIV_IS_20_MIN_LATE', 'mean'),
        pct_cancelled=('DELIV_CANCELLED_AT', lambda x: x.notna().mean()),
        pct_missing_report=('DELIV_MISSING_INCORRECT_REPORT', 'mean'),
        avg_clat=('DELIV_CLAT', 'mean'),
        avg_d2r=('DELIV_D2R', 'mean')
    ).sort_values('num_deliveries', ascending=False)
    store_deliv['pct_late'] = (store_deliv['pct_late'] * 100).round(1)
    store_deliv['pct_cancelled'] = (store_deliv['pct_cancelled'] * 100).round(1)
    store_deliv['pct_missing_report'] = (store_deliv['pct_missing_report'] * 100).round(1)
    store_deliv['avg_clat'] = store_deliv['avg_clat'].round(2)
    store_deliv['avg_d2r'] = store_deliv['avg_d2r'].round(2)
    print(store_deliv.to_string())

def item_analysis(df):
    print("\n" + "=" * 60)
    print("ITEM-LEVEL ANALYSIS")
    print("=" * 60)
    print(f"Total item rows: {len(df):,}")
    print(f"Items found: {df['WAS_FOUND'].sum():,} ({df['WAS_FOUND'].mean()*100:.1f}%)")
    print(f"Items missing: {df['WAS_MISSING'].sum():,} ({df['WAS_MISSING'].mean()*100:.1f}%)")
    print(f"Items substituted: {df['WAS_SUBBED'].sum():,} ({df['WAS_SUBBED'].mean()*100:.1f}%)")

    print("\n--- Category Breakdown ---")
    cat = df.groupby('ITEM_CATEGORY').agg(
        count=('ITEM_NAME', 'count'),
        avg_price=('ITEM_PRICE', 'mean'),
        missing_rate=('WAS_MISSING', 'mean'),
        sub_rate=('WAS_SUBBED', 'mean'),
    ).sort_values('count', ascending=False)
    cat['avg_price'] = cat['avg_price'].round(2)
    cat['missing_rate'] = (cat['missing_rate'] * 100).round(1)
    cat['sub_rate'] = (cat['sub_rate'] * 100).round(1)
    print(cat.head(15).to_string())

def dashmart_vs_grocery(df):
    print("\n" + "=" * 60)
    print("DASHMART vs GROCERY COMPARISON")
    print("=" * 60)
    deliv_items = df.groupby('DELIVERY_UUID').agg(
        total_items=('ITEM_NAME', 'count'),
        missing_count=('WAS_MISSING', 'sum'),
        order_value=('ITEM_PRICE', 'sum'),
        missing_report=('DELIV_MISSING_INCORRECT_REPORT', 'first'),
        store=('DELIV_STORE_NAME', 'first')
    )
    for label, mask in [("DashMart", deliv_items['store'] == 'DashMart1'),
                        ("Grocery (all)", deliv_items['store'].str.startswith('Grocery'))]:
        subset = deliv_items[mask]
        print(f"\n{label}: {len(subset):,} deliveries")
        print(f"  Avg items/order: {subset['total_items'].mean():.1f}")
        print(f"  Avg order value: ${subset['order_value'].mean():.2f}")
        print(f"  Item missing rate: {subset['missing_count'].sum()/subset['total_items'].sum()*100:.1f}%")
        print(f"  Complaint rate: {subset['missing_report'].mean()*100:.1f}%")

if __name__ == '__main__':
    df = load_data()
    deliv = delivery_level_summary(df)
    store_breakdown(deliv)
    item_analysis(df)
    dashmart_vs_grocery(df)
