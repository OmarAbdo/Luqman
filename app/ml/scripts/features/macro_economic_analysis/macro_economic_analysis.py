# macroeconomic_analysis/data_fetcher.py

import os
import requests
import pandas as pd


class MacroeconomicDataFetcher:
    """
    A class to fetch macroeconomic data from the World Bank API.
    """

    BASE_DATA_PATH = "app/ml/data"
    COMPANY_CODE = "AAPL"
    WORLD_BANK_BASE_URL = "http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}?format=json&date={start_year}:{end_year}&per_page=1000"

    INDICATORS = {
        "NY.GDP.MKTP.CD": "Gross Domestic Product (current US$)",
        "FP.CPI.TOTL": "Consumer Price Index (CPI)",
        "SL.UEM.TOTL.ZS": "Unemployment Rate (% of total labor force)",
        "FR.INR.RINR": "Real Interest Rate (%)",
        "GC.DOD.TOTL.GD.ZS": "Government Debt (% of GDP)",
        "PA.NUS.FCRF": "Official Exchange Rate (LCU per US$, period average)",
        "FM.LBL.MQMY.GD.ZS": "Broad Money (M2 as % of GDP)",
        "BN.CAB.XOKA.CD": "Current Account Balance (BoP, current US$)",
        "BX.KLT.DINV.CD.WD": "Foreign Direct Investment (FDI, net inflows, current US$)",
        "NE.EXP.GNFS.CD": "Exports of Goods and Services (current US$)",
    }

    def __init__(self, base_data_path=None, company_code=None):
        self.base_data_path = base_data_path or self.BASE_DATA_PATH
        self.company_code = company_code or self.COMPANY_CODE
        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """
        Ensure the necessary directories exist for storing macroeconomic data.
        """
        company_path = os.path.join(self.base_data_path, self.company_code, "macro")
        if not os.path.exists(company_path):
            os.makedirs(company_path)

    def fetch_world_bank_data(self, country_code, indicator, start_year, end_year):
        """
        Fetch macroeconomic data from the World Bank API.

        :param country_code: The country code (e.g., 'US', 'GB')
        :param indicator: The economic indicator to fetch (e.g., 'NY.GDP.MKTP.CD', 'SL.UEM.TOTL.ZS')
        :param start_year: The start year for the data.
        :param end_year: The end year for the data.
        :return: A DataFrame containing the data.
        """
        url = self.WORLD_BANK_BASE_URL.format(
            country_code=country_code,
            indicator=indicator,
            start_year=start_year,
            end_year=end_year,
        )
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            df = self.process_world_bank_response(data)
            self.save_data(
                df, f"{indicator}_{country_code}_{start_year}_{end_year}_raw"
            )
            return df
        except requests.exceptions.RequestException as e:
            print(
                f"Error fetching data for indicator {indicator} in country {country_code}: {e}"
            )
            return pd.DataFrame()

    def process_world_bank_response(self, data):
        """
        Process the response data from the World Bank API.

        :param data: JSON response data from the World Bank API.
        :return: Processed DataFrame containing the macroeconomic data.
        """
        try:
            if len(data) < 2 or "date" not in data[1][0]:
                print("Unexpected response format from World Bank API.")
                return pd.DataFrame()
            records = data[1]
            df = pd.DataFrame.from_records(records)
            df = df[["country", "indicator", "date", "value"]]
            df.columns = ["Country", "Indicator", "Year", "Value"]
            return df
        except (IndexError, KeyError):
            print("Unexpected response format from World Bank API.")
            return pd.DataFrame()

    def save_data(self, data, filename):
        """
        Save data to a CSV file.

        :param data: DataFrame containing data to save.
        :param filename: Name of the file.
        """
        path = os.path.join(
            self.base_data_path, self.company_code, "macro", f"{filename}.csv"
        )
        data.to_csv(path, index=False)
        print(f"Data saved to {path}")


# Example usage
if __name__ == "__main__":
    fetcher = MacroeconomicDataFetcher()
    country_code = "US"
    start_year = "2010"
    end_year = "2023"

    for indicator, description in MacroeconomicDataFetcher.INDICATORS.items():
        print(f"Fetching data for {description} ({indicator})...")
        data = fetcher.fetch_world_bank_data(
            country_code, indicator, start_year, end_year
        )
        print(data.head())
