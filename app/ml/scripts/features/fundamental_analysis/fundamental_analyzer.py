# scripts/features/fundamental_analysis.py
import yfinance as yf
import pandas as pd
import requests
import json
from financial_metrics import (
    calculate_pe,
    calculate_pb,
    calculate_ev_ebitda,
    calculate_peg_ratio,
    calculate_roe,
    calculate_roa,
    calculate_gross_margin,
    calculate_net_profit_margin,
    calculate_debt_to_equity,
    calculate_interest_coverage_ratio,
    calculate_current_ratio,
    calculate_quick_ratio,
    calculate_revenue_growth,
    calculate_earnings_growth,
    calculate_operating_cash_flow,
    calculate_free_cash_flow,
)


class FundamentalAnalysis:
    """
    A class to perform fundamental analysis by fetching financial metrics from data sources such as Yahoo Finance
    and performing custom calculations if necessary.
    """

    def __init__(self):
        self.ticker = "AAPL"
        self.data = None
        self.financial_data = {}
        self.data_path = "app/ml/data/AAPL"
        self.alpha_vantage_api_key = (
            "YOUR_ALPHA_VANTAGE_API_KEY"  # Replace with your actual API key
        )

    def fetch_yahoo_financials(self):
        """
        Fetch financial data from Yahoo Finance, including income statement, balance sheet, and cash flow.
        """
        stock = yf.Ticker(self.ticker)
        self.financial_data["income_statement"] = stock.financials
        self.financial_data["balance_sheet"] = stock.balance_sheet
        self.financial_data["cash_flow"] = stock.cashflow
        self.financial_data["summary"] = stock.info
        print("Fetched Financial Data:")
        print(self.financial_data)
        with open(f"{self.data_path}/fetched_financial_data.txt", "w") as file:
            file.write(str(self.financial_data))
        return self.financial_data

    def calculate_ratios(self):
        """
        Calculate all fundamental ratios by calling individual methods for each category.
        """
        income_statement = self.financial_data.get("income_statement")
        balance_sheet = self.financial_data.get("balance_sheet")
        cash_flow = self.financial_data.get("cash_flow")
        summary = self.financial_data.get("summary")
        if (
            income_statement is None
            or balance_sheet is None
            or cash_flow is None
            or summary is None
        ):
            raise ValueError(
                "Financial data is incomplete. Make sure to fetch financial data first."
            )

        try:
            # Valuation Ratios
            self.financial_data["P/E"] = calculate_pe(summary)
            self.financial_data["P/B"] = calculate_pb(summary)
            self.financial_data["EV/EBITDA"] = calculate_ev_ebitda(summary)
            self.financial_data["PEG Ratio"] = calculate_peg_ratio(summary)

            # Profitability Ratios
            self.financial_data["ROE"] = calculate_roe(income_statement, balance_sheet)
            self.financial_data["ROA"] = calculate_roa(income_statement, balance_sheet)
            self.financial_data["Gross Margin"] = calculate_gross_margin(
                income_statement
            )
            self.financial_data["Net Profit Margin"] = calculate_net_profit_margin(
                income_statement
            )

            # Leverage Ratios
            self.financial_data["Debt-to-Equity"] = calculate_debt_to_equity(
                balance_sheet
            )
            self.financial_data["Interest Coverage Ratio"] = (
                calculate_interest_coverage_ratio(income_statement)
            )

            # Liquidity Ratios
            self.financial_data["Current Ratio"] = calculate_current_ratio(
                balance_sheet
            )
            self.financial_data["Quick Ratio"] = calculate_quick_ratio(balance_sheet)

            # Growth Metrics
            self.financial_data["Revenue Growth"] = calculate_revenue_growth(
                income_statement
            )
            self.financial_data["Earnings Growth"] = calculate_earnings_growth(
                income_statement
            )

            # Cash Flow Metrics
            self.financial_data["Operating Cash Flow"] = calculate_operating_cash_flow(
                cash_flow
            )
            self.financial_data["Free Cash Flow"] = calculate_free_cash_flow(cash_flow)

        except KeyError as e:
            print(f"KeyError while calculating ratios: {e}")
            # Set missing financial data to None
            for key in [
                "P/E",
                "P/B",
                "EV/EBITDA",
                "PEG Ratio",
                "ROE",
                "ROA",
                "Gross Margin",
                "Net Profit Margin",
                "Debt-to-Equity",
                "Interest Coverage Ratio",
                "Current Ratio",
                "Quick Ratio",
                "Revenue Growth",
                "Earnings Growth",
                "Operating Cash Flow",
                "Free Cash Flow",
            ]:
                if key not in self.financial_data:
                    self.financial_data[key] = None

        with open(f"{self.data_path}/calculated_ratios.txt", "w") as file:
            file.write(str(self.financial_data))

        return self.financial_data

    def fill_gaps_with_alpha_vantage(self):
        """
        Fetch data from Alpha Vantage in case metrics are missing.
        """
        base_url = "https://www.alphavantage.co/query"
        functions = {
            "income_statement": "INCOME_STATEMENT",
            "balance_sheet": "BALANCE_SHEET",
            "cash_flow": "CASH_FLOW",
        }

        for key, function in functions.items():
            if key not in self.financial_data or not self.financial_data[key]:
                params = {
                    "function": function,
                    "symbol": self.ticker,
                    "apikey": self.alpha_vantage_api_key,
                }
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    self.financial_data[key] = response.json()
                else:
                    print(
                        f"Failed to fetch {key} from Alpha Vantage. Status code: {response.status_code}"
                    )

    def get_fundamental_metrics(self):
        """
        Combine all fetched data and calculated metrics into a unified dataframe.
        """
        metrics = {
            "Ticker": self.ticker,
            "P/E": self.financial_data.get("P/E"),
            "P/B": self.financial_data.get("P/B"),
            "EV/EBITDA": self.financial_data.get("EV/EBITDA"),
            "PEG Ratio": self.financial_data.get("PEG Ratio"),
            "ROE": self.financial_data.get("ROE"),
            "ROA": self.financial_data.get("ROA"),
            "Gross Margin": self.financial_data.get("Gross Margin"),
            "Net Profit Margin": self.financial_data.get("Net Profit Margin"),
            "Debt-to-Equity": self.financial_data.get("Debt-to-Equity"),
            "Interest Coverage Ratio": self.financial_data.get(
                "Interest Coverage Ratio"
            ),
            "Current Ratio": self.financial_data.get("Current Ratio"),
            "Quick Ratio": self.financial_data.get("Quick Ratio"),
            "Revenue Growth": self.financial_data.get("Revenue Growth"),
            "Earnings Growth": self.financial_data.get("Earnings Growth"),
            "Operating Cash Flow": self.financial_data.get("Operating Cash Flow"),
            "Free Cash Flow": self.financial_data.get("Free Cash Flow"),
        }
        return pd.DataFrame([metrics])


if __name__ == "__main__":
    fundamental_analysis = FundamentalAnalysis()
    fetched_data = fundamental_analysis.fetch_yahoo_financials()
    fundamental_analysis.fill_gaps_with_alpha_vantage()
    calculated_data = fundamental_analysis.calculate_ratios()
    print("Calculated Financial Ratios:")
    print(calculated_data)
