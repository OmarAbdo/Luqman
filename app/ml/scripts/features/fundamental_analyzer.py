# scripts/features/fundamental_analysis.py
import yfinance as yf
import pandas as pd


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
        # print(self.financial_data)
        
        with open(f"{self.data_path}/fetched_financial_data.txt", "w") as file:
            file.write(str(self.financial_data))
            print("financial data saved to fetched_financial_data.txt")
        return self.financial_data

    def calculate_valuation_ratios(self, summary):
        """
        Calculate valuation ratios such as P/E, P/B, EV/EBITDA, and PEG Ratio.
        """
        self.financial_data["P/E"] = summary.get("trailingPE", None)
        self.financial_data["P/B"] = summary.get("priceToBook", None)
        self.financial_data["EV/EBITDA"] = summary.get("enterpriseToEbitda", None)
        self.financial_data["PEG Ratio"] = summary.get("pegRatio", None)

    def calculate_profitability_ratios(self, income_statement, balance_sheet):
        """
        Calculate profitability ratios such as ROE, ROA, Gross Margin, and Net Profit Margin.
        """
        net_income = (
            income_statement.loc["Net Income"].iloc[0]
            if "Net Income" in income_statement.index
            else None
        )
        total_equity = (
            balance_sheet.loc["Total Stockholder Equity"].iloc[0]
            if "Total Stockholder Equity" in balance_sheet.index
            else None
        )
        total_assets = (
            balance_sheet.loc["Total Assets"].iloc[0]
            if "Total Assets" in balance_sheet.index
            else None
        )
        gross_profit = (
            income_statement.loc["Gross Profit"].iloc[0]
            if "Gross Profit" in income_statement.index
            else None
        )
        revenue = (
            income_statement.loc["Total Revenue"].iloc[0]
            if "Total Revenue" in income_statement.index
            else None
        )

        self.financial_data["ROE"] = (
            net_income / total_equity if total_equity and total_equity != 0 else None
        )
        self.financial_data["ROA"] = (
            net_income / total_assets if total_assets and total_assets != 0 else None
        )
        self.financial_data["Gross Margin"] = (
            gross_profit / revenue if revenue and revenue != 0 else None
        )
        self.financial_data["Net Profit Margin"] = (
            net_income / revenue if revenue and revenue != 0 else None
        )

    def calculate_leverage_ratios(self, income_statement, balance_sheet):
        """
        Calculate leverage ratios such as Debt-to-Equity and Interest Coverage Ratio.
        """
        total_liabilities = (
            balance_sheet.loc["Total Liab"].iloc[0]
            if "Total Liab" in balance_sheet.index
            else None
        )
        interest_expense = (
            income_statement.loc["Interest Expense"].iloc[0]
            if "Interest Expense" in income_statement.index
            else None
        )
        ebit = (
            income_statement.loc["Ebit"].iloc[0]
            if "Ebit" in income_statement.index
            else None
        )

        total_equity = (
            balance_sheet.loc["Total Stockholder Equity"].iloc[0]
            if "Total Stockholder Equity" in balance_sheet.index
            else None
        )
        self.financial_data["Debt-to-Equity"] = (
            total_liabilities / total_equity
            if total_equity and total_equity != 0
            else None
        )
        self.financial_data["Interest Coverage Ratio"] = (
            ebit / interest_expense
            if ebit is not None
            and interest_expense is not None
            and interest_expense != 0
            else None
        )

    def calculate_liquidity_ratios(self, balance_sheet):
        """
        Calculate liquidity ratios such as Current Ratio and Quick Ratio.
        """
        current_assets = (
            balance_sheet.loc["Total Current Assets"].iloc[0]
            if "Total Current Assets" in balance_sheet.index
            else None
        )
        current_liabilities = (
            balance_sheet.loc["Total Current Liabilities"].iloc[0]
            if "Total Current Liabilities" in balance_sheet.index
            else None
        )

        self.financial_data["Current Ratio"] = (
            current_assets / current_liabilities
            if current_liabilities and current_liabilities != 0
            else None
        )
        self.financial_data["Quick Ratio"] = (
            (current_assets - balance_sheet.loc["Inventory"].iloc[0])
            / current_liabilities
            if current_liabilities
            and current_liabilities != 0
            and "Inventory" in balance_sheet.index
            else None
        )

    def calculate_growth_metrics(self, income_statement):
        """
        Calculate growth metrics such as Revenue Growth and Earnings Growth.
        """
        revenue = (
            income_statement.loc["Total Revenue"].iloc[0]
            if "Total Revenue" in income_statement.index
            else None
        )
        previous_revenue = (
            income_statement.loc["Total Revenue"].iloc[1]
            if "Total Revenue" in income_statement.index
            and len(income_statement.loc["Total Revenue"]) > 1
            else None
        )
        net_income = (
            income_statement.loc["Net Income"].iloc[0]
            if "Net Income" in income_statement.index
            else None
        )
        previous_net_income = (
            income_statement.loc["Net Income"].iloc[1]
            if "Net Income" in income_statement.index
            and len(income_statement.loc["Net Income"]) > 1
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

    def calculate_cash_flow_metrics(self, cash_flow):
        """
        Calculate cash flow metrics such as Operating Cash Flow and Free Cash Flow.
        """
        operating_cash_flow = (
            cash_flow.loc["Total Cash From Operating Activities"].iloc[0]
            if "Total Cash From Operating Activities" in cash_flow.index
            else None
        )
        capex = (
            cash_flow.loc["Capital Expenditures"].iloc[0]
            if "Capital Expenditures" in cash_flow.index
            else 0
        )

        self.financial_data["Operating Cash Flow"] = operating_cash_flow
        self.financial_data["Free Cash Flow"] = (
            operating_cash_flow - capex if capex else None
        )

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
            self.calculate_valuation_ratios(summary)
            self.calculate_profitability_ratios(income_statement, balance_sheet)
            self.calculate_leverage_ratios(income_statement, balance_sheet)
            self.calculate_liquidity_ratios(balance_sheet)
            self.calculate_growth_metrics(income_statement)
            self.calculate_cash_flow_metrics(cash_flow)

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
            print("calculated ratios saved to calculated_ratios.txt")

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
    fundamental_analysis = FundamentalAnalysis()
    fetched_data = fundamental_analysis.fetch_yahoo_financials()
    calculated_data = fundamental_analysis.calculate_ratios()
    # print("Calculated Financial Ratios:")
    # print(calculated_data)
