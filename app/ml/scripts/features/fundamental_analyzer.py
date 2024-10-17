# scripts/features/fundamental_analysis.py
import yfinance as yf
import pandas as pd


class FundamentalAnalysis:
    """
    A class to perform fundamental analysis by fetching financial metrics from data sources such as Yahoo Finance
    and performing custom calculations if necessary.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = None
        self.financial_data = {}

    def fetch_yahoo_financials(self):
        """
        Fetch financial data from Yahoo Finance, including income statement, balance sheet, and cash flow.
        """
        stock = yf.Ticker(self.ticker)
        self.financial_data["income_statement"] = stock.financials
        self.financial_data["balance_sheet"] = stock.balance_sheet
        self.financial_data["cash_flow"] = stock.cashflow
        self.financial_data["summary"] = stock.info
        return self.financial_data

    def calculate_ratios(self):
        """
        Calculate fundamental ratios like P/E, P/B, ROE, etc. from the financial statements.
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
            self.financial_data["P/E"] = summary.get("trailingPE", None)
            self.financial_data["P/B"] = summary.get("priceToBook", None)
            self.financial_data["EV/EBITDA"] = summary.get("enterpriseToEbitda", None)
            self.financial_data["PEG Ratio"] = summary.get("pegRatio", None)

            # Profitability Ratios
            net_income = income_statement.loc["Net Income"][0]
            total_equity = balance_sheet.loc["Total Stockholder Equity"][0]
            total_assets = balance_sheet.loc["Total Assets"][0]
            gross_profit = income_statement.loc["Gross Profit"][0]
            revenue = income_statement.loc["Total Revenue"][0]

            self.financial_data["ROE"] = (
                net_income / total_equity if total_equity != 0 else None
            )
            self.financial_data["ROA"] = (
                net_income / total_assets if total_assets != 0 else None
            )
            self.financial_data["Gross Margin"] = (
                gross_profit / revenue if revenue != 0 else None
            )
            self.financial_data["Net Profit Margin"] = (
                net_income / revenue if revenue != 0 else None
            )

            # Leverage Ratios
            total_liabilities = balance_sheet.loc["Total Liab"][0]
            interest_expense = income_statement.loc.get("Interest Expense", [None])[0]
            ebit = income_statement.loc["Ebit"][0]

            self.financial_data["Debt-to-Equity"] = (
                total_liabilities / total_equity if total_equity != 0 else None
            )
            self.financial_data["Interest Coverage Ratio"] = (
                ebit / interest_expense
                if interest_expense and interest_expense != 0
                else None
            )

            # Liquidity Ratios
            current_assets = balance_sheet.loc["Total Current Assets"][0]
            current_liabilities = balance_sheet.loc["Total Current Liabilities"][0]

            self.financial_data["Current Ratio"] = (
                current_assets / current_liabilities
                if current_liabilities != 0
                else None
            )
            self.financial_data["Quick Ratio"] = (
                (current_assets - balance_sheet.loc["Inventory"][0])
                / current_liabilities
                if current_liabilities != 0
                else None
            )

            # Growth Metrics
            previous_revenue = (
                income_statement.loc["Total Revenue"][1]
                if len(income_statement.loc["Total Revenue"]) > 1
                else None
            )
            previous_net_income = (
                income_statement.loc["Net Income"][1]
                if len(income_statement.loc["Net Income"]) > 1
                else None
            )

            self.financial_data["Revenue Growth"] = (
                ((revenue - previous_revenue) / previous_revenue)
                if previous_revenue
                else None
            )
            self.financial_data["Earnings Growth"] = (
                ((net_income - previous_net_income) / previous_net_income)
                if previous_net_income
                else None
            )

            # Cash Flow Metrics
            operating_cash_flow = cash_flow.loc["Total Cash From Operating Activities"][
                0
            ]
            capex = cash_flow.loc.get("Capital Expenditures", [0])[0]

            self.financial_data["Operating Cash Flow"] = operating_cash_flow
            self.financial_data["Free Cash Flow"] = (
                operating_cash_flow - capex if capex else None
            )

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

        return self.financial_data

    def fill_gaps_with_alpha_vantage(self):
        """
        Placeholder for fetching data from Alpha Vantage in case metrics are missing.
        """
        # Future implementation for filling gaps using Alpha Vantage API
        pass

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
    # Example usage
    ticker_symbol = "AAPL"
    fundamental_analysis = FundamentalAnalysis(ticker_symbol)
    financial_data = fundamental_analysis.fetch_yahoo_financials()
    calculated_data = fundamental_analysis.calculate_ratios()
    metrics = fundamental_analysis.get_fundamental_metrics()
    print(metrics)
