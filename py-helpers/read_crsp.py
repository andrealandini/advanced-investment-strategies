import pandas as pd
from pathlib import Path

def preview_and_save_heads():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    files = [
        ("crsp_daily_named.txt", "crsp_daily_named_sample.txt"),
        ("crsp_monthly_named.txt", "crsp_monthly_named_sample.txt")
    ]

    for full, sample in files:
        # Prefer the big file, fallback to the sample
        if (data_dir / full).exists():
            path = data_dir / full
            label = full
        elif (data_dir / sample).exists():
            path = data_dir / sample
            label = sample
        else:
            print(f"\n❌ Neither {full} nor {sample} found.")
            continue

        print(f"\nProcessing {label}:\n" + "-" * 50)
        try:
            df = pd.read_csv(path, sep="\t")

            # Take head
            df_head = df.head()

            # Save to new file with "_head" suffix
            out_path = path.with_name(path.stem + "_head.txt")
            df_head.to_csv(out_path, sep="\t", index=False)

            print(f"✅ Saved head to {out_path.name}")
        except Exception as e:
            print(f"⚠️ Error reading {label}: {e}")

if __name__ == "__main__":
    preview_and_save_heads()

