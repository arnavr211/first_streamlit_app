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

def lateness_analysis(df):
    """Lateness analysis with timestamps converted from UTC to Eastern Time (Cincinnati)."""
    deliv = df.groupby('DELIVERY_UUID').first()
    deliv['created_local'] = deliv['DELIV_CREATED_AT'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    deliv['hour_local'] = deliv['created_local'].dt.hour
    deliv['dow_local'] = deliv['created_local'].dt.day_name()

    print("\n" + "=" * 60)
    print("HOUR-OF-DAY DELIVERY PATTERNS (Eastern Time / Local)")
    print("=" * 60)
    hour = deliv.groupby('hour_local').agg(
        deliveries=('DELIV_CREATED_AT', 'count'),
        pct_late=('DELIV_IS_20_MIN_LATE', 'mean'),
        avg_clat=('DELIV_CLAT', 'mean')
    ).sort_index()
    hour['pct_late'] = (hour['pct_late'] * 100).round(1)
    hour['avg_clat'] = hour['avg_clat'].round(2)
    print(hour.to_string())

    print("\n" + "=" * 60)
    print("DAYPART ANALYSIS (Eastern Time / Local)")
    print("=" * 60)
    def daypart(h):
        if 6 <= h < 11: return '1-Morning (6am-11am)'
        elif 11 <= h < 14: return '2-Lunch (11am-2pm)'
        elif 14 <= h < 17: return '3-Afternoon (2pm-5pm)'
        elif 17 <= h < 21: return '4-Dinner (5pm-9pm)'
        elif 21 <= h <= 23: return '5-Late Night (9pm-12am)'
        else: return '6-Overnight (12am-6am)'

    deliv['daypart'] = deliv['hour_local'].apply(daypart)
    dp = deliv.groupby('daypart').agg(
        deliveries=('DELIV_CREATED_AT', 'count'),
        pct_late=('DELIV_IS_20_MIN_LATE', 'mean'),
        avg_clat=('DELIV_CLAT', 'mean'),
        pct_cancelled=('DELIV_CANCELLED_AT', lambda x: x.notna().mean()),
        pct_complaint=('DELIV_MISSING_INCORRECT_REPORT', 'mean')
    ).sort_index()
    dp['pct_late'] = (dp['pct_late'] * 100).round(1)
    dp['avg_clat'] = dp['avg_clat'].round(2)
    dp['pct_cancelled'] = (dp['pct_cancelled'] * 100).round(1)
    dp['pct_complaint'] = (dp['pct_complaint'] * 100).round(1)
    print(dp.to_string())

    print("\n" + "=" * 60)
    print("LATE RATE BY STORE x DAYPART (Eastern Time)")
    print("=" * 60)
    sp = deliv.groupby(['DELIV_STORE_NAME', 'daypart']).agg(
        deliveries=('DELIV_CREATED_AT', 'count'),
        pct_late=('DELIV_IS_20_MIN_LATE', 'mean')
    )
    sp['pct_late'] = (sp['pct_late'] * 100).round(1)
    print(sp.to_string())


if __name__ == '__main__':
    df = load_data()
    deliv = delivery_level_summary(df)
    store_breakdown(deliv)
    item_analysis(df)
    dashmart_vs_grocery(df)
    lateness_analysis(df)
