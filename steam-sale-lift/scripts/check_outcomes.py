import pandas as pd
tc = pd.read_parquet("results/eda_treatment_control.parquet")
print(tc.groupby("treated")[["pre_reviews","sale_reviews","post_reviews","log_lift"]].mean().round(3))
print(f"\nN treated: {tc['treated'].sum()}  N control: {(~tc['treated']).sum()}")
print(f"\nMean log_lift treated: {tc[tc['treated']]['log_lift'].mean():.3f}")
print(f"Mean log_lift control: {tc[~tc['treated']]['log_lift'].mean():.3f}")
print(f"Difference (raw DiD estimate): {tc[tc['treated']]['log_lift'].mean() - tc[~tc['treated']]['log_lift'].mean():.3f}")
