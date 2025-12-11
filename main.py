import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def load_series(path, label):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    def parse_num(x):
        if isinstance(x, str):
            x = x.replace(',', '')
        return float(x)

    df[label] = df["Price"].apply(parse_num)
    df = df[["Date", label]].sort_values("Date").reset_index(drop=True)
    return df


moex_w = load_series("/MOEX 2 yrs weekly.csv", "MOEX")
moex_m = load_series("/MOEX 5 yrs monthly.csv", "MOEX")
mts_w = load_series("/MTS 2 yrs weekly.csv", "MTS")
mts_m = load_series("/MTS 5 yrs monthly.csv", "MTS")


def prepare_returns(idx_df, stock_df):
    merged = pd.merge(stock_df, idx_df, on="Date", how="inner").sort_values("Date")
    merged["ret_stock"] = merged["MTS"].pct_change()
    merged["ret_mkt"] = merged["MOEX"].pct_change()
    merged = merged.dropna().reset_index(drop=True)
    return merged


ret_w = prepare_returns(moex_w, mts_w)
ret_m = prepare_returns(moex_m, mts_m)


def regress_beta(ret_df):
    X = sm.add_constant(ret_df["ret_mkt"])
    y = ret_df["ret_stock"]
    model = sm.OLS(y, X).fit()
    beta = model.params["ret_mkt"]
    alpha = model.params["const"]
    r2 = model.rsquared
    return beta, alpha, r2, model


beta_w, alpha_w, r2_w, model_w = regress_beta(ret_w)
beta_m, alpha_m, r2_m, model_m = regress_beta(ret_m)

print("Weekly (2y) beta:", beta_w, "alpha:", alpha_w, "R^2:", r2_w)
print("Monthly (5y) beta:", beta_m, "alpha:", alpha_m, "R^2:", r2_m)

# Normalized price dynamics, weekly (2y)
plt.figure()
for df, label in [(moex_w, "MOEX (2y, weekly)"), (mts_w, "MTS (2y, weekly)")]:
    norm = df.set_index("Date") / df.set_index("Date").iloc[0]
    plt.plot(norm.index, norm.iloc[:, 0], label=label)
plt.title("Normalized prices: MTS vs MOEX (2 years, weekly)")
plt.xlabel("Date")
plt.ylabel("Index (start = 1)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Normalized price dynamics, monthly (5y)
plt.figure()
for df, label in [(moex_m, "MOEX (5y, monthly)"), (mts_m, "MTS (5y, monthly)")]:
    norm = df.set_index("Date") / df.set_index("Date").iloc[0]
    plt.plot(norm.index, norm.iloc[:, 0], label=label)
plt.title("Normalized prices: MTS vs MOEX (5 years, monthly)")
plt.xlabel("Date")
plt.ylabel("Index (start = 1)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter with regression line, weekly
plt.figure()
plt.scatter(ret_w["ret_mkt"], ret_w["ret_stock"])
x_vals = pd.Series(sorted(ret_w["ret_mkt"]))
y_fit = alpha_w + beta_w * x_vals
plt.plot(x_vals, y_fit)
plt.title("MTS vs MOEX weekly returns (2y)")
plt.xlabel("MOEX return")
plt.ylabel("MTS return")
plt.tight_layout()
plt.show()

# Scatter with regression line, monthly
plt.figure()
plt.scatter(ret_m["ret_mkt"], ret_m["ret_stock"])
x_vals_m = pd.Series(sorted(ret_m["ret_mkt"]))
y_fit_m = alpha_m + beta_m * x_vals_m
plt.plot(x_vals_m, y_fit_m)
plt.title("MTS vs MOEX monthly returns (5y)")
plt.xlabel("MOEX return")
plt.ylabel("MTS return")
plt.tight_layout()
plt.show()
