import os
import argparse
import pandas as pd
from dotenv import load_dotenv

# GA4 Data API (v1beta)
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    RunReportRequest, Dimension, Metric, Filter, FilterExpression
)

METRICS = [
    "screenPageViews",
    "activeUsers",
    "newUsers",
    "sessions",
    "engagedSessions",
    "engagementRate",
    "averageSessionDuration",
    "eventCount"
]

def load_urls(path: str) -> list[str]:
    urls: list[str] = []
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, header=None)
        urls = [str(x).strip() for x in df.iloc[:, 0].tolist() if str(x).strip() and str(x) != "nan"]
    else:
        with open(path, "r", encoding="utf-8") as f:
            urls = [l.strip() for l in f if l.strip()]
    return [u for u in dict.fromkeys(urls) if u]  # dedup keep order

def fetch_ga4(property_id: str, urls: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    client = BetaAnalyticsDataClient()
    rows = []
    BATCH = 50
    for i in range(0, len(urls), BATCH):
        batch = urls[i:i + BATCH]
        req = RunReportRequest(
            property=f"properties/{property_id}",
            dimensions=[Dimension(name="pageLocation"), Dimension(name="pageTitle")],
            metrics=[Metric(name=m) for m in METRICS],
            date_ranges=[{"start_date": start_date, "end_date": end_date}],
            dimension_filter=FilterExpression(
                filter=Filter(
                    field_name="pageLocation",
                    in_list_filter=Filter.InListFilter(values=batch)
                )
            ),
            limit=100000
        )
        resp = client.run_report(req)
        for row in resp.rows:
            rec = {
                "pageLocation": row.dimension_values[0].value,
                "pageTitle": row.dimension_values[1].value,
            }
            for j, m in enumerate(METRICS):
                rec[m] = row.metric_values[j].value
            rows.append(rec)

    df = pd.DataFrame(rows)

    missing = [u for u in urls if df.empty or u not in set(df.get("pageLocation", pd.Series(dtype=str)).tolist())]
    if missing:
        zero_row = {m: 0 for m in METRICS}
        zero_row.update({"pageTitle": "", "pageLocation": None})
        zeros = pd.DataFrame([dict(zero_row, pageLocation=u) for u in missing])
        df = pd.concat([df, zeros], ignore_index=True)

    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df

def main():
    load_dotenv()
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    parser = argparse.ArgumentParser(description="Export GA4 metrics by list of URLs into CSV")
    parser.add_argument("--property", default=os.getenv("GA4_PROPERTY_ID", ""), help="GA4 Property ID (или из .env)")
    parser.add_argument("--urls", required=True, help="Путь к .txt или .csv (1 URL в строке)")
    parser.add_argument("--start", default="30daysAgo", help="Дата начала (YYYY-MM-DD или 30daysAgo)")
    parser.add_argument("--end", default="today", help="Дата конца (YYYY-MM-DD или today)")
    parser.add_argument("--out", default="ga4_by_urls.csv", help="Путь к результирующему CSV")
    args = parser.parse_args()

    if not args.property:
        raise SystemExit("Не указан GA4 Property ID (--property или GA4_PROPERTY_ID в .env)")
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise SystemExit("Не найдена переменная GOOGLE_APPLICATION_CREDENTIALS (путь к JSON-ключу). Проверьте .env")

    urls = load_urls(args.urls)
    if not urls:
        raise SystemExit("Файл со ссылками пуст или нечитабелен.")

    df = fetch_ga4(args.property, urls, args.start, args.end)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Готово. Строк: {len(df)} → {args.out}")

if __name__ == "__main__":
    main()
