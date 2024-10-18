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
    
    ## Overview of Workflow
    1. **Data Fetching**:
    - Initially, financial data is fetched from Yahoo Finance, using the `fetch_yahoo_financials()` method.
    - Missing or incomplete data points are then filled by using Alpha Vantage API, which we implemented in the `fill_gaps_with_alpha_vantage()` method.

    2. **Financial Ratios and Metrics Calculation**:
    - The class calculates various financial metrics, such as **valuation ratios**, **profitability ratios**, **leverage ratios**, and more, using the `calculate_ratios()` method.
    - For better code readability and modularity, we separated individual metric calculations into a custom library (`financial_metrics.py`). Each calculation function handles a specific metric and is reusable for different tickers or data sets.

    3. **Integration and Enhancement**:
    - The `perform_analysis()` method integrates the workflow to facilitate data fetching, gap filling, ratio calculation, and preparation of the output in one streamlined function.
    - We recently added the ability to generate a text file of calculated metrics (`final_metrics.txt`). However, there is also room for further optimization.

    ## Potential Improvements
    1. **Conversion from Text File to CSV**:
    - While the text file is helpful for initial testing and logging, converting the metrics storage into CSV files would have numerous benefits:
        - CSV files allow for easy data integration in other parts of the project without having to re-run API calls or recompute metrics.
        - Using CSV files can save computational power and API limits, as Alpha Vantage has strict call limits, and we can cache results in the CSV for reuse.

    2. **Data Normalization and Feature Preparation**:
    - Adding a `prepare_features()` method within the class would be useful. This method would include data cleaning, normalization, and scaling of calculated metrics, converting them into a ready-to-use format for model training (e.g., LSTM).

    3. **Support for New Data Sources**:
    - To improve the availability of missing metrics like the **Current Ratio** and **Quick Ratio**, we may integrate more data sources or use web scraping tools to collect financial data that isn’t accessible via Yahoo Finance or Alpha Vantage.

    4. **Logging Mechanism**:
    - A more sophisticated logging mechanism can be implemented. Currently, the workflow generates text-based output for validation, but a logging library can help track activities, especially errors or missing data, in a systematic way.

    ## Integration with Feature Engineering Class
    - Our goal is to ultimately use the `FundamentalAnalyzer` as part of a more extensive feature engineering process. To facilitate this, `perform_analysis()` provides a DataFrame containing the final metrics, which can be fed directly into our feature engineering workflows.
    - The future `prepare_features()` method will handle scaling and normalizing the metrics so that they are in a usable format for models, ensuring consistency between fundamental and technical features.
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
        # print(self.financial_data)
        with open(f"{self.data_path}/fetched_financial_data.txt", "w") as file:
            file.write(str(self.financial_data))
            print("file written: ", file)
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

        print("calculate_ratios")
        with open(f"{self.data_path}/calculated_ratios.txt", "w") as file:
            file.write(str(self.financial_data))
            print("file written: ", file)

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
            if key not in self.financial_data or self._is_data_missing(
                self.financial_data[key]
            ):
                params = {
                    "function": function,
                    "symbol": self.ticker,
                    "apikey": self.alpha_vantage_api_key,
                }
                response = requests.get(base_url, params=params)
                if response.status_code == 200:
                    try:
                        data = pd.DataFrame(response.json())
                        if not data.empty:
                            self.financial_data[key] = data
                    except ValueError:
                        print(f"Unexpected data format for {key} from Alpha Vantage.")
                else:
                    print(
                        f"Failed to fetch {key} from Alpha Vantage. Status code: {response.status_code}"
                    )

    def _is_data_missing(self, data):
        """
        Helper function to determine if financial data is missing or incomplete.
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return (
                data.isna().all().all()
                if isinstance(data, pd.DataFrame)
                else data.isna().all()
            )
        return True  # If data is not a DataFrame/Series, assume it's missing.

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

    def perform_analysis(self):
        """
        Perform the full workflow of fundamental analysis:
        1. Fetch financial data from Yahoo Finance.
        2. Fill any gaps in the data using Alpha Vantage.
        3. Calculate financial ratios and metrics.
        4. Prepare the metrics for further use or integration.
        """
        self.fetch_yahoo_financials()
        self.fill_gaps_with_alpha_vantage()
        self.calculate_ratios()
        fundamental_metrics_df = self.get_fundamental_metrics()
        with open(f"{self.data_path}/final_metrics.txt", "w") as file:
            file.write(str(fundamental_metrics_df.to_string()))
        print("file written: ", file)
        print("Completed Fundamental Analysis.")

        return fundamental_metrics_df


if __name__ == "__main__":
    fundamental_analysis = FundamentalAnalysis()
    final_metrics = fundamental_analysis.perform_analysis()
    print(final_metrics)
