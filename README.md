# Retirement Income Optimizer

This is a Streamlit app preloaded with your retirement assumptions:
- Married couple retiring at age 62
- Delayed Social Security to age 70
- Tax-deferred, taxable, 457(b), and 457(f) balances
- 457(f) lump sum in 2033
- Inheritance in 2035
- Beach house sale at age 85
- 6% average return, 12% volatility, 3% inflation
- Decaying spending path with a 10% downturn guardrail
- Monte Carlo success target

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Put `app.py`, `requirements.txt`, and `README.md` in a GitHub repo.
2. In Streamlit Community Cloud, create a new app from that repo.
3. Select `app.py` as the entrypoint.
4. Open the resulting URL on your iPad and use Safari > Share > Add to Home Screen.

## Notes

This app is a planning tool. It does not replace a CPA, tax attorney, estate lawyer, or fiduciary advisor.
