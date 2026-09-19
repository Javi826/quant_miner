import pandas as pd
from pathlib import Path

FOLDER = Path("/home/javi/Escritorio/hacienda")
FORMATTED = FOLDER / "formatted"
YEAR = 2026


def convert_xls_to_xlsx(path: Path) -> Path:
    FORMATTED.mkdir(exist_ok=True)
    out_path = FORMATTED / path.with_suffix(".xlsx").name
    df = pd.read_excel(path, engine="xlrd")
    df.to_excel(out_path, index=False)
    return out_path


def inspect_excel(path: Path) -> None:
    df = pd.read_excel(path, nrows=0)
    print(f"\n{'=' * 60}")
    print(f"File: {path.name}")
    print(f"{'=' * 60}")
    for i, col in enumerate(df.columns, start=1):
        print(f"  {i:>3}. {col}")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace("USDT", "", regex=False).str.strip(),
        errors="coerce",
    )


def analyze_pnl(path: Path, year: int = YEAR) -> None:
    df = pd.read_excel(path, parse_dates=["Closed time"])
    df_2025 = df[df["Closed time"].dt.year == year].copy()

    if df_2025.empty:
        print("\nNo positions closed in 2025.")
        return

    gross_pnl = clean_numeric(df_2025["Realized PnL"]).sum()
    total_fees = clean_numeric(df_2025["Fees"]).sum()
    net_pnl = gross_pnl - total_fees

    print(f"\n{'=' * 60}")
    print(f"{year} P&L Summary (posiciones_futuros)")
    print(f"{'=' * 60}")
    print(f"  Positions closed : {len(df_2025)}")
    print(f"  Gross PnL        : {gross_pnl:.2f} USDT")
    print(f"  Total fees       : {total_fees:.2f} USDT")
    print(f"  Net PnL          : {net_pnl:.2f} USDT")


def main() -> None:
    files = sorted(FOLDER.glob("*.xls"))
    if not files:
        print(f"No .xls files found in {FOLDER}")
        return

    converted = []
    for file in files:
        xlsx_path = convert_xls_to_xlsx(file)
        converted.append(xlsx_path)
        print(f"Converted: {file.name} -> formatted/{xlsx_path.name}")

    for file in converted:
        inspect_excel(file)

    print(f"\nTotal files: {len(converted)}")

    posiciones_path = FORMATTED / "posiciones_futuros_USDTM.xlsx"
    if posiciones_path.exists():
        analyze_pnl(posiciones_path)


if __name__ == "__main__":
    main()