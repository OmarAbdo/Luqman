import unittest
import pandas as pd
import numpy as np
from app.ml.scripts.components.fundamental_analysis.financial_metrics import (
    calculate_net_profit_margin,
    calculate_roa,
    calculate_gross_margin,
    calculate_debt_to_equity,
    calculate_interest_coverage_ratio,
    calculate_operating_cash_flow,
    calculate_free_cash_flow,
    calculate_revenue_growth,
    calculate_earnings_growth,
)


class TestFinancialMetrics(unittest.TestCase):

    def setUp(self):
        self.income_statement = pd.DataFrame(
            {
                "2023-09-30": [1000000, 400000, 150000, 250000],
                "2022-09-30": [900000, 350000, 120000, 220000],
            },
            index=["Total Revenue", "Net Income", "Gross Profit", "EBIT"],
        )

        self.balance_sheet = pd.DataFrame(
            {
                "2023-09-30": [500000, 200000, 100000],
                "2022-09-30": [450000, 180000, 95000],
            },
            index=["Total Assets", "Total Debt", "Total Stockholder Equity"],
        )

        self.cash_flow = pd.DataFrame(
            {"2023-09-30": [300000, 200000], "2022-09-30": [280000, 180000]},
            index=["Operating Cash Flow", "Free Cash Flow"],
        )

    def test_net_profit_margin(self):
        result = calculate_net_profit_margin(self.income_statement)
        expected = (
            self.income_statement.loc["Net Income"].iloc[0]
            / self.income_statement.loc["Total Revenue"].iloc[0]
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_return_on_assets(self):
        result = calculate_roa(self.income_statement, self.balance_sheet)
        expected = (
            self.income_statement.loc["Net Income"].iloc[0]
            / self.balance_sheet.loc["Total Assets"].iloc[0]
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_gross_margin(self):
        result = calculate_gross_margin(self.income_statement)
        expected = (
            self.income_statement.loc["Gross Profit"].iloc[0]
            / self.income_statement.loc["Total Revenue"].iloc[0]
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_debt_to_equity_ratio(self):
        result = calculate_debt_to_equity(self.balance_sheet)
        expected = (
            self.balance_sheet.loc["Total Debt"].iloc[0]
            / self.balance_sheet.loc["Total Stockholder Equity"].iloc[0]
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_interest_coverage_ratio(self):
        self.income_statement.loc["Interest Expense"] = [50000, 45000]
        result = calculate_interest_coverage_ratio(self.income_statement)
        expected = (
            self.income_statement.loc["EBIT"].iloc[0]
            / self.income_statement.loc["Interest Expense"].iloc[0]
        )
        self.assertAlmostEqual(result, expected, places=5)

    def test_operating_cash_flow(self):
        result = calculate_operating_cash_flow(self.cash_flow)
        expected = self.cash_flow.loc["Operating Cash Flow"].iloc[0]
        self.assertEqual(result, expected)

    def test_free_cash_flow(self):
        result = calculate_free_cash_flow(self.cash_flow)
        expected = self.cash_flow.loc["Free Cash Flow"].iloc[0]
        self.assertEqual(result, expected)

    def test_revenue_growth(self):
        result = calculate_revenue_growth(self.income_statement)
        expected = (
            self.income_statement.loc["Total Revenue"].iloc[0]
            - self.income_statement.loc["Total Revenue"].iloc[1]
        ) / self.income_statement.loc["Total Revenue"].iloc[1]
        self.assertAlmostEqual(result, expected, places=5)

    def test_net_income_growth(self):
        result = calculate_earnings_growth(self.income_statement)
        expected = (
            self.income_statement.loc["Net Income"].iloc[0]
            - self.income_statement.loc["Net Income"].iloc[1]
        ) / self.income_statement.loc["Net Income"].iloc[1]
        self.assertAlmostEqual(result, expected, places=5)


if __name__ == "__main__":
    unittest.main()
