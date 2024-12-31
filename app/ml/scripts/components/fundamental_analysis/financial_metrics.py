# financial_metrics_library.py


# Valuation Ratios:
# Price-to-Earnings (P/E), Price-to-Book (P/B), Enterprise Value to EBITDA (EV/EBITDA), and PEG Ratio are crucial for comparing valuation across companies.
def calculate_pe(summary):
    try:
        return summary["trailingPE"]
    except KeyError:
        return None


# Price-to-Book (P/B) Ratio
# Measures a company's market value relative to its book value, indicating whether a stock is over- or under-valued.
def calculate_pb(summary):
    try:
        return summary["priceToBook"]
    except KeyError:
        return None


# Enterprise Value to EBITDA (EV/EBITDA)
# Useful for comparing companies with different capital structures, as it considers both debt and equity.
def calculate_ev_ebitda(summary):
    try:
        return summary["enterpriseToEbitda"]
    except KeyError:
        return None


# PEG Ratio
# Evaluates a company's valuation considering its growth rate, giving a more complete picture of value.
def calculate_peg_ratio(summary):
    try:
        return summary["pegRatio"]
    except KeyError:
        return None


# Profitability Ratios:
# Return on Equity (ROE), Return on Assets (ROA), Gross Margin, and Net Profit Margin show how efficiently a company generates profits.


def calculate_roe(income_statement, balance_sheet):
    try:
        net_income = income_statement.loc["Net Income"].iloc[0]
        total_equity = balance_sheet.loc["Total Stockholder Equity"].iloc[0]
        return net_income / total_equity if total_equity != 0 else None
    except KeyError:
        return None


# Return on Assets (ROA)
# Indicates how effectively a company uses its assets to generate profit.
def calculate_roa(income_statement, balance_sheet):
    try:
        net_income = income_statement.loc["Net Income"].iloc[0]
        total_assets = balance_sheet.loc["Total Assets"].iloc[0]
        return net_income / total_assets if total_assets != 0 else None
    except KeyError:
        return None


# Gross Margin
# Measures the percentage of revenue that exceeds the cost of goods sold, showing the financial success of production and sales.
def calculate_gross_margin(income_statement):
    try:
        gross_profit = income_statement.loc["Gross Profit"].iloc[0]
        revenue = income_statement.loc["Total Revenue"].iloc[0]
        return gross_profit / revenue if revenue != 0 else None
    except KeyError:
        return None


# Net Profit Margin
# Represents the percentage of revenue that results in profit, reflecting overall efficiency and profitability.
def calculate_net_profit_margin(income_statement):
    try:
        net_income = income_statement.loc["Net Income"].iloc[0]
        revenue = income_statement.loc["Total Revenue"].iloc[0]
        return net_income / revenue if revenue != 0 else None
    except KeyError:
        return None


# Leverage Ratios:
# Debt-to-Equity and Interest Coverage Ratios indicate how a company finances its operations and its ability to meet financial obligations.


def calculate_debt_to_equity(balance_sheet):
    try:
        total_debt = balance_sheet.loc["Total Debt"].iloc[0]
        total_equity = balance_sheet.loc["Total Stockholder Equity"].iloc[0]
        return total_debt / total_equity if total_equity != 0 else None
    except KeyError:
        return None


# Interest Coverage Ratio
# Shows how well a company can pay interest on outstanding debt, indicating financial stability.
def calculate_interest_coverage_ratio(income_statement):
    try:
        ebit = income_statement.loc["EBIT"].iloc[0]
        interest_expense = income_statement.loc["Interest Expense"].iloc[0]
        return ebit / interest_expense if interest_expense != 0 else None
    except KeyError:
        return None


# Liquidity Ratios:
# Current and Quick Ratios show a company's ability to cover its short-term liabilities.


def calculate_current_ratio(balance_sheet):
    try:
        current_assets = balance_sheet.loc["Total Current Assets"].iloc[0]
        current_liabilities = balance_sheet.loc["Total Current Liabilities"].iloc[0]
        return (
            current_assets / current_liabilities if current_liabilities != 0 else None
        )
    except KeyError:
        return None


# Quick Ratio
# Measures liquidity excluding inventory, giving a more conservative view of a company's ability to meet short-term liabilities.
def calculate_quick_ratio(balance_sheet):
    try:
        current_assets = balance_sheet.loc["Total Current Assets"].iloc[0]
        inventory = balance_sheet.loc["Inventory"].iloc[0]
        current_liabilities = balance_sheet.loc["Total Current Liabilities"].iloc[0]
        return (
            (current_assets - inventory) / current_liabilities
            if current_liabilities != 0
            else None
        )
    except KeyError:
        return None


# Growth Metrics:
# Revenue and Earnings Growth indicate how fast the company is growing, which is essential for evaluating future prospects.


def calculate_revenue_growth(income_statement):
    try:
        revenue_current = income_statement.loc["Total Revenue"].iloc[0]
        revenue_previous = income_statement.loc["Total Revenue"].iloc[1]
        return (
            (revenue_current - revenue_previous) / revenue_previous
            if revenue_previous != 0
            else None
        )
    except KeyError:
        return None


# Earnings Growth
# Measures growth in net income, which is important for assessing a company's increasing profitability.
def calculate_earnings_growth(income_statement):
    try:
        net_income_current = income_statement.loc["Net Income"].iloc[0]
        net_income_previous = income_statement.loc["Net Income"].iloc[1]
        return (
            (net_income_current - net_income_previous) / net_income_previous
            if net_income_previous != 0
            else None
        )
    except KeyError:
        return None


# Cash Flow Metrics:
# Operating Cash Flow and Free Cash Flow represent the cash a company generates and the cash available after capital expenditures.


def calculate_operating_cash_flow(cash_flow):
    try:
        return cash_flow.loc["Operating Cash Flow"].iloc[0]
    except KeyError:
        return None


# Free Cash Flow
# Indicates the cash a company has after paying for operational and capital expenditures, crucial for assessing its financial flexibility.
def calculate_free_cash_flow(cash_flow):
    try:
        return cash_flow.loc["Free Cash Flow"].iloc[0]
    except KeyError:
        return None
