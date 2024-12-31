# macroeconomic_analysis/data_fetcher.py

import os
import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class MacroeconomicDataFetcher:
    """
    A class to fetch macroeconomic data from the World Bank API and calculate derived metrics.
    """

    BASE_DATA_PATH = "app/luqman/data"
    COMPANY_CODE = os.getenv("TICKER")
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
            if not df.empty:
                df = self.normalize_data(df)
                self.save_data(
                    df, f"{indicator}_{country_code}_{start_year}_{end_year}_normalized"
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
            df["Indicator"] = df["Indicator"].apply(
                lambda x: str(x)
            )  # Ensure all indicators are strings
            return df
        except (IndexError, KeyError):
            print("Unexpected response format from World Bank API.")
            return pd.DataFrame()

    def normalize_data(self, df):
        """
        Normalize the data using Min-Max scaling.

        :param df: DataFrame containing the data to normalize.
        :return: Normalized DataFrame.
        """
        scaler = MinMaxScaler()
        df["Value"] = scaler.fit_transform(df[["Value"]])
        return df

    def calculate_derived_metrics(self, df):
        """
        Calculate derived metrics such as growth rates and ratios.

        :param df: DataFrame containing the raw macroeconomic data.
        :return: DataFrame containing the derived metrics.
        """
        derived_metrics = []
        indicators = df["Indicator"].unique()
        for indicator in indicators:
            indicator_df = df[df["Indicator"] == indicator].copy()
            indicator_df.sort_values(by="Year", inplace=True)
            if indicator == "NY.GDP.MKTP.CD":
                # Calculate GDP Growth Rate
                indicator_df["GDP Growth Rate"] = indicator_df["Value"].pct_change()
            elif indicator == "FP.CPI.TOTL":
                # Calculate Inflation Rate
                indicator_df["Inflation Rate"] = indicator_df["Value"].pct_change()
            elif indicator == "GC.DOD.TOTL.GD.ZS" and "NY.GDP.MKTP.CD" in indicators:
                # Calculate Debt-to-GDP Ratio
                gdp_df = df[df["Indicator"] == "NY.GDP.MKTP.CD"].copy()
                gdp_df = gdp_df[["Country", "Year", "Value"]].rename(
                    columns={"Value": "GDP_Value"}
                )
                merged_df = pd.merge(
                    indicator_df, gdp_df, on=["Country", "Year"], how="inner"
                )
                indicator_df["Debt-to-GDP Ratio"] = (
                    merged_df["Value"] / merged_df["GDP_Value"]
                )
            derived_metrics.append(indicator_df)
        if derived_metrics:
            return pd.concat(derived_metrics, ignore_index=True)
        return df

    def prepare_features(self, df):
        """
        Prepare the macroeconomic metrics for model training by normalizing and scaling the data.

        :param df: DataFrame containing the derived metrics.
        :return: Normalized and scaled DataFrame ready for model training.
        """
        scaler = MinMaxScaler()
        feature_columns = [
            col for col in df.columns if col not in ["Country", "Indicator", "Year"]
        ]
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        return df

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

    all_data = []
    for indicator, description in MacroeconomicDataFetcher.INDICATORS.items():
        print(f"Fetching data for {description} ({indicator})...")
        data = fetcher.fetch_world_bank_data(
            country_code, indicator, start_year, end_year
        )
        if not data.empty:
            all_data.append(data)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        derived_data = fetcher.calculate_derived_metrics(combined_data)
        final_features = fetcher.prepare_features(derived_data)
        fetcher.save_data(
            final_features, f"macro_features_{country_code}_{start_year}_{end_year}"
        )
