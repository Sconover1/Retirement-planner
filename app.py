import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Retirement Optimizer", layout="wide")

# -----------------------------
# Defaults tailored to user inputs
# -----------------------------
DEFAULTS = {
    "current_year": 2026,
    "retirement_year": 2028,
    "owner_birth_year": 1966,
    "owner_birth_month": 6,
    "owner_birth_day": 21,
    "spouse_birth_year": 1968,
    "spouse_birth_month": 1,
    "spouse_birth_day": 9,
    "retire_age": 62,
    "end_age": 93,
    "ss_claim_age": 70,
    "tax_deferred": 5_400_000.0,
    "acct_457b": 500_000.0,
    "acct_457f": 750_000.0,
    "taxable": 3_800_000.0,
    "cash": 200_000.0,
    "roth": 0.0,
    "inheritance_year": 2035,
    "inheritance_amount": 1_500_000.0,
    "beach_value": 1_850_000.0,
    "beach_mortgage_payoff": 300_000.0,
    "beach_appreciation": 0.02,
    "beach_sale_age": 85,
    "primary_residence": 800_000.0,
    "primary_home_appreciation": 0.02,
    "return_mean": 0.06,
    "return_std": 0.12,
    "inflation": 0.03,
    "simulation_count": 750,
    "success_target": 0.90,
    "ss_owner_monthly": 5_181.0,
    "ss_spouse_monthly": 5_181.0,
    "effective_taxable_cost_basis_ratio": 0.75,  # portion of taxable withdrawal treated as basis
    "state_tax_rate": 0.0575,  # MD approximate state income tax placeholder; editable
    "starting_standard_deduction": 30_000.0,  # MFJ 2025
    "ordinary_brackets_mfj": [
        (23_850.0, 0.10),
        (96_950.0, 0.12),
        (206_700.0, 0.22),
        (394_600.0, 0.24),
        (501_050.0, 0.32),
        (751_600.0, 0.35),
        (float("inf"), 0.37),
    ],
    "cap_gains_thresholds_mfj": [
        (98_900.0, 0.00),
        (613_700.0, 0.15),
        (float("inf"), 0.20),
    ],
}


def age_on_jan1(year: int, birth_year: int) -> int:
    return year - birth_year - 1


def age_midyear(year: int, birth_year: int) -> int:
    return year - birth_year


def infl_adj(base: float, years: int, inflation: float) -> float:
    return base * ((1 + inflation) ** max(0, years))


def tax_from_brackets(taxable_income: float, brackets: List[Tuple[float, float]]) -> float:
    tax = 0.0
    prev = 0.0
    income = max(0.0, taxable_income)
    for upper, rate in brackets:
        if income <= prev:
            break
        taxed_here = min(income, upper) - prev
        if taxed_here > 0:
            tax += taxed_here * rate
        prev = upper
    return tax


def cap_gains_tax(ordinary_taxable: float, taxable_gains: float, thresholds: List[Tuple[float, float]]) -> float:
    """Approximate preferential LTCG tax by stacking gains on top of ordinary taxable income."""
    remaining_gains = max(0.0, taxable_gains)
    tax = 0.0
    prev = 0.0
    base = max(0.0, ordinary_taxable)
    for upper, rate in thresholds:
        band_start = max(base, prev)
        band_room = max(0.0, upper - band_start)
        taxed_here = min(remaining_gains, band_room)
        tax += taxed_here * rate
        remaining_gains -= taxed_here
        prev = upper
        if remaining_gains <= 0:
            break
    if remaining_gains > 0:
        tax += remaining_gains * thresholds[-1][1]
    return tax


def social_security_taxable(combined_income: float, annual_benefit: float) -> float:
    """Approximate MFJ Social Security taxation."""
    if annual_benefit <= 0:
        return 0.0
    if combined_income <= 32_000:
        return 0.0
    if combined_income <= 44_000:
        return min(0.5 * (combined_income - 32_000), 0.5 * annual_benefit)
    base = min(6_000.0, 0.5 * annual_benefit)
    extra = 0.85 * (combined_income - 44_000)
    return min(0.85 * annual_benefit, base + extra)


def compute_federal_tax(
    ordinary_income_before_ss: float,
    annual_ss: float,
    taxable_cap_gains: float,
    standard_deduction: float,
    ordinary_brackets: List[Tuple[float, float]],
    cap_gains_thresholds: List[Tuple[float, float]],
) -> Dict[str, float]:
    combined_income = ordinary_income_before_ss + taxable_cap_gains + 0.5 * annual_ss
    ss_taxable = social_security_taxable(combined_income, annual_ss)
    ordinary_total = ordinary_income_before_ss + ss_taxable

    deduction_left = standard_deduction
    ordinary_taxable = max(0.0, ordinary_total - deduction_left)
    deduction_left = max(0.0, deduction_left - ordinary_total)
    cap_gains_taxable = max(0.0, taxable_cap_gains - deduction_left)

    fed_ordinary = tax_from_brackets(ordinary_taxable, ordinary_brackets)
    fed_capgains = cap_gains_tax(ordinary_taxable, cap_gains_taxable, cap_gains_thresholds)

    return {
        "ss_taxable": ss_taxable,
        "ordinary_taxable": ordinary_taxable,
        "cap_gains_taxable": cap_gains_taxable,
        "federal_tax": fed_ordinary + fed_capgains,
    }


@dataclass
class YearResult:
    year: int
    owner_age: int
    spouse_age: int
    spending_target: float
    spending_cut_applied: float
    social_security: float
    gross_spending_need: float
    roth_conversion: float
    taxable_withdrawal: float
    taxable_gains_realized: float
    tax_deferred_withdrawal: float
    acct_457b_withdrawal: float
    roth_withdrawal: float
    ordinary_income_before_ss: float
    federal_tax: float
    state_tax: float
    total_tax: float
    end_taxable: float
    end_tax_deferred: float
    end_457b: float
    end_roth: float
    net_worth_spendable: float
    beach_home_value: float
    primary_home_value: float
    portfolio_peak: float
    guardrail_active: bool


def determine_base_spending(owner_age: int, inflation_years: int, inflation: float) -> float:
    base = 450_000.0 if owner_age <= 66 else 430_000.0
    if owner_age <= 75:
        decay = 1.0
    elif owner_age <= 85:
        decay = 0.9
    else:
        decay = 0.8
    return infl_adj(base * decay, inflation_years, inflation)


def approximate_rmd(balance: float, age: int) -> float:
    # Rough Uniform Lifetime divisors for ages 73-95; sufficient for planning model.
    divisors = {
        73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
        80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
        87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
        94: 9.5, 95: 8.9,
    }
    if age < 73:
        return 0.0
    div = divisors.get(age, 8.1)
    return max(0.0, balance / div)


def run_single_path(config: Dict, returns: np.ndarray, optimize: bool = True) -> Tuple[bool, List[YearResult]]:
    retirement_year = config["retirement_year"]
    end_year = retirement_year + (config["end_age"] - config["retire_age"])

    taxable = config["taxable"] + config["cash"]
    tax_deferred = config["tax_deferred"]
    acct_457b = config["acct_457b"]
    roth = config["roth"]
    peak = taxable + tax_deferred + acct_457b + roth

    results: List[YearResult] = []

    for i, year in enumerate(range(retirement_year, end_year + 1)):
        owner_age = age_midyear(year, config["owner_birth_year"])
        spouse_age = age_midyear(year, config["spouse_birth_year"])
        infl_years = year - retirement_year

        base_spending = determine_base_spending(owner_age, infl_years, config["inflation"])

        # Dynamic spending guardrail: if portfolio is below trailing peak, cut up to 10%
        spendable_now = taxable + tax_deferred + acct_457b + roth
        drawdown = 0.0 if peak <= 0 else max(0.0, 1 - spendable_now / peak)
        cut_pct = min(0.10, max(0.0, (drawdown - 0.10) / 0.20 * 0.10)) if drawdown > 0.10 else 0.0
        spending_target = base_spending * (1 - cut_pct)
        guardrail_active = cut_pct > 0.0

        annual_ss = 0.0
        if owner_age >= config["ss_claim_age"]:
            annual_ss += 12 * infl_adj(config["ss_owner_monthly"], year - config["current_year"], 0.02)
        if spouse_age >= config["ss_claim_age"]:
            annual_ss += 12 * infl_adj(config["ss_spouse_monthly"], year - config["current_year"], 0.02)

        year_457f = config["retirement_year"] + 5  # June 2033 relative to June 2028 retirement
        forced_457f_income = config["acct_457f"] if year == year_457f else 0.0
        inheritance = config["inheritance_amount"] if year == config["inheritance_year"] else 0.0

        beach_value = config["beach_value"] * ((1 + config["beach_appreciation"]) ** (year - retirement_year))
        primary_value = config["primary_residence"] * ((1 + config["primary_home_appreciation"]) ** (year - retirement_year))
        beach_sale_proceeds = 0.0
        if owner_age == config["beach_sale_age"]:
            beach_sale_proceeds = max(0.0, beach_value - config["beach_mortgage_payoff"])

        # Approximate basis ratio on taxable withdrawals to estimate realized gains.
        basis_ratio = min(max(config["effective_taxable_cost_basis_ratio"], 0.0), 1.0)

        # Spending to fund net of Social Security.
        net_cash_needed = max(0.0, spending_target - annual_ss)

        # RMD approximations. Assume first spouse RMD begins when each reaches 73.
        owner_rmd = approximate_rmd(0.55 * tax_deferred, owner_age)
        spouse_rmd = approximate_rmd(0.45 * tax_deferred, spouse_age)
        rmd = min(tax_deferred, owner_rmd + spouse_rmd)

        # Optimizer: avoid Roth conversions in 457(f) spike year and one year around it.
        roth_conversion = 0.0
        if optimize:
            target_nominal_rate = 0.24
            std_ded = infl_adj(config["starting_standard_deduction"], infl_years, config["inflation"])
            ordinary_brackets = [(infl_adj(u, infl_years, config["inflation"]), r) for u, r in config["ordinary_brackets_mfj"]]
            top_24 = next(u for u, r in ordinary_brackets if abs(r - target_nominal_rate) < 1e-9)
            ordinary_base = forced_457f_income + rmd
            if year not in {year_457f - 1, year_457f, year_457f + 1} and owner_age < 73 and drawdown < 0.20:
                room = max(0.0, top_24 + std_ded - ordinary_base)
                roth_conversion = min(room * 0.35, tax_deferred * 0.05)

        # Initial cash sourcing preference: taxable, 457b, RMD/IRA, Roth last.
        taxable_wd = min(taxable, net_cash_needed * 0.65)
        remaining_need = net_cash_needed - taxable_wd

        acct_457b_wd = min(acct_457b, remaining_need * 0.5)
        remaining_need -= acct_457b_wd

        tax_deferred_wd = min(tax_deferred, max(rmd, remaining_need))
        remaining_need -= min(remaining_need, tax_deferred_wd)

        roth_wd = min(roth, max(0.0, remaining_need))
        remaining_need -= roth_wd

        # If still short, tap whichever account remains with priority taxable then tax deferred.
        if remaining_need > 1:
            extra_taxable = min(taxable - taxable_wd, remaining_need)
            taxable_wd += max(0.0, extra_taxable)
            remaining_need -= max(0.0, extra_taxable)
        if remaining_need > 1:
            extra_td = min(tax_deferred - tax_deferred_wd - roth_conversion, remaining_need)
            tax_deferred_wd += max(0.0, extra_td)
            remaining_need -= max(0.0, extra_td)
        if remaining_need > 1:
            return False, results

        # Tax estimate, iterative-ish one pass with taxes added to withdrawals.
        taxable_gains = taxable_wd * (1 - basis_ratio) + max(0.0, beach_sale_proceeds * 0.70)
        ordinary_before_ss = forced_457f_income + acct_457b_wd + tax_deferred_wd + roth_conversion
        std_ded = infl_adj(config["starting_standard_deduction"], infl_years, config["inflation"])
        ordinary_brackets = [(infl_adj(u, infl_years, config["inflation"]), r) for u, r in config["ordinary_brackets_mfj"]]
        cap_thresholds = [(infl_adj(u, infl_years, config["inflation"]), r) for u, r in config["cap_gains_thresholds_mfj"]]

        tax_info = compute_federal_tax(ordinary_before_ss, annual_ss, taxable_gains, std_ded, ordinary_brackets, cap_thresholds)
        state_tax = config["state_tax_rate"] * (ordinary_before_ss + 0.5 * annual_ss + 0.25 * taxable_gains)
        total_tax = tax_info["federal_tax"] + state_tax

        # Cover taxes using taxable first; if insufficient, use tax-deferred, then Roth.
        tax_from_taxable = min(max(0.0, taxable - taxable_wd), total_tax)
        taxable_wd += tax_from_taxable
        taxable_gains += tax_from_taxable * (1 - basis_ratio)
        remaining_tax = total_tax - tax_from_taxable

        if remaining_tax > 0:
            extra_td_for_tax = min(max(0.0, tax_deferred - tax_deferred_wd - roth_conversion), remaining_tax)
            tax_deferred_wd += extra_td_for_tax
            ordinary_before_ss += extra_td_for_tax
            remaining_tax -= extra_td_for_tax
        if remaining_tax > 0:
            extra_roth_for_tax = min(roth, remaining_tax)
            roth_wd += extra_roth_for_tax
            remaining_tax -= extra_roth_for_tax
        if remaining_tax > 1:
            return False, results

        # Recompute taxes after extra withdrawals.
        tax_info = compute_federal_tax(ordinary_before_ss, annual_ss, taxable_gains, std_ded, ordinary_brackets, cap_thresholds)
        state_tax = config["state_tax_rate"] * (ordinary_before_ss + 0.5 * annual_ss + 0.25 * taxable_gains)
        total_tax = tax_info["federal_tax"] + state_tax

        # Apply flows before growth.
        taxable = taxable - taxable_wd + inheritance + beach_sale_proceeds
        tax_deferred = tax_deferred - tax_deferred_wd - roth_conversion
        acct_457b = acct_457b - acct_457b_wd
        roth = roth - roth_wd + roth_conversion

        if min(taxable, tax_deferred, acct_457b, roth) < -1e-6:
            return False, results

        # Growth applied to investable balances only.
        r = returns[i]
        taxable *= (1 + r)
        tax_deferred *= (1 + r)
        acct_457b *= (1 + r)
        roth *= (1 + r)

        if any(x < -1 for x in [taxable, tax_deferred, acct_457b, roth]):
            return False, results

        spendable = taxable + tax_deferred + acct_457b + roth
        peak = max(peak, spendable)

        results.append(
            YearResult(
                year=year,
                owner_age=owner_age,
                spouse_age=spouse_age,
                spending_target=spending_target,
                spending_cut_applied=cut_pct,
                social_security=annual_ss,
                gross_spending_need=net_cash_needed,
                roth_conversion=roth_conversion,
                taxable_withdrawal=taxable_wd,
                taxable_gains_realized=taxable_gains,
                tax_deferred_withdrawal=tax_deferred_wd,
                acct_457b_withdrawal=acct_457b_wd,
                roth_withdrawal=roth_wd,
                ordinary_income_before_ss=ordinary_before_ss,
                federal_tax=tax_info["federal_tax"],
                state_tax=state_tax,
                total_tax=total_tax,
                end_taxable=taxable,
                end_tax_deferred=tax_deferred,
                end_457b=acct_457b,
                end_roth=roth,
                net_worth_spendable=spendable,
                beach_home_value=0.0 if owner_age >= config["beach_sale_age"] else beach_value,
                primary_home_value=primary_value,
                portfolio_peak=peak,
                guardrail_active=guardrail_active,
            )
        )

    success = len(results) == (end_year - retirement_year + 1)
    return success, results


def run_monte_carlo(config: Dict, optimize: bool = True):
    n_years = config["end_age"] - config["retire_age"] + 1
    n_sims = int(config["simulation_count"])
    success = 0
    terminal_values = []
    sampled_paths = []

    rng = np.random.default_rng(42)
    for s in range(n_sims):
        returns = rng.normal(config["return_mean"], config["return_std"], size=n_years)
        ok, path = run_single_path(config, returns, optimize=optimize)
        if ok:
            success += 1
            terminal_values.append(path[-1].net_worth_spendable)
        else:
            terminal_values.append(0.0)
        if s < 40 and path:
            sampled_paths.append([p.net_worth_spendable for p in path])

    prob = success / n_sims if n_sims else 0.0
    n_years_actual = len(sampled_paths[0]) if sampled_paths else n_years
    years = list(range(config["retirement_year"], config["retirement_year"] + n_years_actual))
    sample_df = pd.DataFrame(sampled_paths, columns=years) if sampled_paths else pd.DataFrame()

    # Deterministic baseline using mean returns
    baseline_ok, baseline_results = run_single_path(config, np.array([config["return_mean"]] * n_years), optimize=optimize)
    baseline_df = pd.DataFrame([r.__dict__ for r in baseline_results]) if baseline_results else pd.DataFrame()

    return {
        "success_probability": prob,
        "terminal_values": np.array(terminal_values),
        "sample_df": sample_df,
        "baseline_ok": baseline_ok,
        "baseline_df": baseline_df,
    }


# -----------------------------
# UI
# -----------------------------
st.title("Retirement Income Optimizer")
st.caption("Tax-aware Monte Carlo retirement planner with spending guardrails, 457(f) spike handling, inheritance, and property events.")

with st.sidebar:
    st.header("Plan inputs")
    config = DEFAULTS.copy()
    config["simulation_count"] = st.slider("Monte Carlo simulations", 250, 3000, DEFAULTS["simulation_count"], step=250)
    config["success_target"] = st.slider("Required confidence of success", 0.70, 0.99, DEFAULTS["success_target"], step=0.01)
    config["return_mean"] = st.number_input("Average annual investment return", 0.00, 0.15, DEFAULTS["return_mean"], step=0.005, format="%.3f")
    config["return_std"] = st.number_input("Annual return volatility", 0.01, 0.30, DEFAULTS["return_std"], step=0.005, format="%.3f")
    config["inflation"] = st.number_input("Annual inflation", 0.00, 0.10, DEFAULTS["inflation"], step=0.0025, format="%.3f")
    config["state_tax_rate"] = st.number_input("State/local income tax rate", 0.00, 0.15, DEFAULTS["state_tax_rate"], step=0.0025, format="%.4f")
    st.divider()
    config["tax_deferred"] = st.number_input("Tax-deferred accounts at retirement", 0.0, 20_000_000.0, DEFAULTS["tax_deferred"], step=100_000.0)
    config["acct_457b"] = st.number_input("457(b) at retirement", 0.0, 5_000_000.0, DEFAULTS["acct_457b"], step=50_000.0)
    config["acct_457f"] = st.number_input("457(f) lump sum in 2033", 0.0, 5_000_000.0, DEFAULTS["acct_457f"], step=50_000.0)
    config["taxable"] = st.number_input("Taxable investments at retirement", 0.0, 20_000_000.0, DEFAULTS["taxable"], step=100_000.0)
    config["cash"] = st.number_input("Added cash at retirement", 0.0, 5_000_000.0, DEFAULTS["cash"], step=25_000.0)
    config["roth"] = st.number_input("Roth balance at retirement", 0.0, 20_000_000.0, DEFAULTS["roth"], step=50_000.0)
    st.divider()
    config["inheritance_amount"] = st.number_input("Inheritance amount", 0.0, 10_000_000.0, DEFAULTS["inheritance_amount"], step=50_000.0)
    config["inheritance_year"] = st.number_input("Inheritance year", 2028, 2050, DEFAULTS["inheritance_year"], step=1)
    config["beach_value"] = st.number_input("Beach house value at retirement", 0.0, 10_000_000.0, DEFAULTS["beach_value"], step=50_000.0)
    config["beach_mortgage_payoff"] = st.number_input("Beach house mortgage/payoff at sale", 0.0, 5_000_000.0, DEFAULTS["beach_mortgage_payoff"], step=25_000.0)
    config["beach_appreciation"] = st.number_input("Beach house annual appreciation", 0.00, 0.10, DEFAULTS["beach_appreciation"], step=0.0025, format="%.3f")
    config["beach_sale_age"] = st.number_input("Beach house sale age", 62, 100, DEFAULTS["beach_sale_age"], step=1)
    config["primary_residence"] = st.number_input("Primary residence current value", 0.0, 10_000_000.0, DEFAULTS["primary_residence"], step=50_000.0)
    config["primary_home_appreciation"] = st.number_input("Primary residence annual appreciation", 0.00, 0.10, DEFAULTS["primary_home_appreciation"], step=0.0025, format="%.3f")
    st.divider()
    config["ss_owner_monthly"] = st.number_input("Owner monthly Social Security at 70", 0.0, 10_000.0, DEFAULTS["ss_owner_monthly"], step=50.0)
    config["ss_spouse_monthly"] = st.number_input("Spouse monthly Social Security at 70", 0.0, 10_000.0, DEFAULTS["ss_spouse_monthly"], step=50.0)
    config["effective_taxable_cost_basis_ratio"] = st.slider("Taxable withdrawal basis share", 0.0, 1.0, DEFAULTS["effective_taxable_cost_basis_ratio"], step=0.05)

results = run_monte_carlo(config, optimize=True)
prob = results["success_probability"]
base = results["baseline_df"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Success probability", f"{prob:.1%}")
col2.metric("Target", f"{config['success_target']:.0%}")
if len(results["terminal_values"]):
    median_terminal = float(np.median(results["terminal_values"]))
    p10_terminal = float(np.percentile(results["terminal_values"], 10))
else:
    median_terminal = 0.0
    p10_terminal = 0.0
col3.metric("Median terminal investable wealth", f"${median_terminal:,.0f}")
col4.metric("10th percentile terminal wealth", f"${p10_terminal:,.0f}")

if prob >= config["success_target"]:
    st.success("This plan clears the selected confidence threshold under the model assumptions.")
else:
    st.warning("This plan falls short of the selected confidence threshold. The quickest levers are lower spending, lower tax drag, or more flexibility during downturns.")

if not base.empty:
    st.subheader("Deterministic baseline path (using average return every year)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(base["year"], base["end_taxable"], label="Taxable")
    ax1.plot(base["year"], base["end_tax_deferred"], label="Tax-deferred")
    ax1.plot(base["year"], base["end_457b"], label="457(b)")
    ax1.plot(base["year"], base["end_roth"], label="Roth")
    ax1.plot(base["year"], base["net_worth_spendable"], label="Investable total", linewidth=2)
    ax1.set_ylabel("Dollars")
    ax1.set_xlabel("Year")
    ax1.legend()
    ax1.grid(alpha=0.25)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(base["year"], base["spending_target"], label="Spending target")
    ax2.plot(base["year"], base["social_security"], label="Social Security")
    ax2.bar(base["year"], base["roth_conversion"], label="Roth conversion", alpha=0.5)
    ax2.bar(base["year"], base["total_tax"], label="Total tax", alpha=0.5)
    ax2.set_ylabel("Dollars")
    ax2.set_xlabel("Year")
    ax2.legend()
    ax2.grid(alpha=0.25)
    st.pyplot(fig2)

if not results["sample_df"].empty:
    st.subheader("Sample Monte Carlo paths")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for _, row in results["sample_df"].iterrows():
        ax3.plot(results["sample_df"].columns, row.values, alpha=0.18)
    ax3.set_ylabel("Investable wealth")
    ax3.set_xlabel("Year")
    ax3.grid(alpha=0.25)
    st.pyplot(fig3)

if not base.empty:
    display_cols = [
        "year", "owner_age", "spouse_age", "spending_target", "spending_cut_applied",
        "social_security", "roth_conversion", "taxable_withdrawal", "tax_deferred_withdrawal",
        "acct_457b_withdrawal", "roth_withdrawal", "total_tax", "end_taxable",
        "end_tax_deferred", "end_457b", "end_roth", "net_worth_spendable",
        "beach_home_value", "primary_home_value", "guardrail_active"
    ]
    st.subheader("Year-by-year baseline table")
    show_df = base[display_cols].copy()
    st.dataframe(show_df.style.format({c: "${:,.0f}" for c in show_df.columns if c not in ["year", "owner_age", "spouse_age", "guardrail_active"]}).format({"spending_cut_applied": "{:.1%}"}), use_container_width=True)

    # Export CSV
    csv_bytes = show_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download baseline CSV", data=csv_bytes, file_name="retirement_optimizer_baseline.csv", mime="text/csv")

st.subheader("Model notes")
st.markdown(
    """
- This is a planning model, not tax or legal advice.
- Federal tax defaults are editable and seeded from recent MFJ federal parameters; state tax is a simple editable rate.
- Taxable withdrawals use a user-set basis share to approximate realized gains.
- The 457(f) amount is modeled as ordinary income in 2033.
- The optimizer avoids Roth conversions around the 457(f) spike year, then resumes them selectively before RMD years.
- Spending can be reduced by up to 10% after prolonged drawdowns, per your instructions.
- Primary residence is included in net worth views but not used to fund spending.
"""
)
