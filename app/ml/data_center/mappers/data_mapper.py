# data_center/mappers/data_mapper.py


class DataMapper:
    def __init__(self):
        """
        Initialize the DataMapper class.
        Responsible for mapping raw data from different sources to standardized format.
        """
        pass

    def map_yahoo_data(self, raw_data):
        """
        Map raw Yahoo data to standardized format.

        :param raw_data: Raw data fetched from Yahoo Finance
        :return: Mapped data in standardized format
        """
        try:
            mapped_data = {
                "Total Revenue": (
                    raw_data["income_statement"].loc["Total Revenue"].iloc[0]
                    if "income_statement" in raw_data
                    else None
                ),
                "Net Income": (
                    raw_data["income_statement"].loc["Net Income"].iloc[0]
                    if "income_statement" in raw_data
                    else None
                ),
                "Total Assets": (
                    raw_data["balance_sheet"].loc["Total Assets"].iloc[0]
                    if "balance_sheet" in raw_data
                    else None
                ),
                "EBIT": (
                    raw_data["income_statement"].loc["EBIT"].iloc[0]
                    if "income_statement" in raw_data
                    else None
                ),
                "Free Cash Flow": (
                    raw_data["cash_flow"].loc["Free Cash Flow"].iloc[0]
                    if "cash_flow" in raw_data
                    else None
                ),
            }
            return pd.DataFrame([mapped_data])
        except Exception as e:
            print(f"Error mapping Yahoo data: {e}")
            return pd.DataFrame()

    def map_alpha_vantage_data(self, raw_data):
        """
        Map raw Alpha Vantage data to standardized format.

        :param raw_data: Raw data fetched from Alpha Vantage
        :return: Mapped data in standardized format
        """
        try:
            mapped_data = {
                "Total Revenue": raw_data.get("TotalRevenueTTM"),
                "Net Income": raw_data.get("NetIncomeTTM"),
                "Total Assets": raw_data.get("TotalAssets"),
                "EBIT": raw_data.get("EBIT"),
                "Free Cash Flow": raw_data.get("FreeCashFlowTTM"),
            }
            return pd.DataFrame([mapped_data])
        except Exception as e:
            print(f"Error mapping Alpha Vantage data: {e}")
            return pd.DataFrame()
