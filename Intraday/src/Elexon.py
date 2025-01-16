import requests
import pandas as pd
import datetime as dtt
from abc import ABC, abstractmethod


# Define an interface for the data downloaders
class MarketDataDownloader(ABC):
    @abstractmethod
    def updata_blob_data(self):
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass


class ELEXONDownloader(MarketDataDownloader, ABC, path_to_source="/home/dana/development/intradays/Intraday/sources"):
    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

    def __init__(self):
        self.self.path_to_source = "data"
        pass

    @abstractmethod
    def get_dataset_name(self):
        pass

    @abstractmethod
    def updata_blob_data(self):
        pass

    def build_url(self, endpoint):
        return f"{self.BASE_URL}/{endpoint}"

    def make_request(self, url, params):
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', data)  # Some endpoints return data directly
        else:
            raise Exception(f"Failed to download data: {response.status_code} - {response.text}")


class ATLDownloader(ELEXONDownloader):
    """
    Actual total load (ATL/B0610)
    This endpoint provides actual total load data per bidding zone. It can be filtered by settlement period dates.
    This API endpoint has a maximum range of 7 days.
    """

    def __init__(self, start_date, end_date):
        super().__init__()
        self.dataset_name = "Actual Total Load (ATL)"
        self.data = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.url = f"{self.BASE_URL}/demand/actual/total"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        # Implement the data download and processing logic
        params = {
            "from": self.start_date.strftime("%Y-%m-%d"),
            "to": self.end_date.strftime("%Y-%m-%d"),
            "format": "json",
        }

        response = requests.get(self.url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            # Save to CSV or upload to Azure Blob Storage
            df.to_csv(
                f"{self.path_to_source}/{self.dataset_name.replace(' ', '_')}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv",
                index=False,
            )
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
        else:
            print(f"Failed to download data: {response.status_code}")


class DATLDownloader(ELEXONDownloader):
    """
    Day-Ahead Total Load Forecast Per Bidding Zone (DATL / B0620)
    This endpoint provides day-ahead total load forecast per bidding zone data.
    This API endpoint provides a maximum data output range of 7 days.
    """

    def __init__(self, publish_datetime_from, publish_datetime_to):
        super().__init__()
        self.dataset_name = "Day-Ahead Total Load Forecast (DATL)"
        self.data = pd.DataFrame()
        self.publish_datetime_from = publish_datetime_from
        self.publish_datetime_to = publish_datetime_to
        self.url = f"{self.BASE_URL}/datasets/DATL"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            "publishDateTimeFrom": self.publish_datetime_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "publishDateTimeTo": self.publish_datetime_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json",
        }

        response = requests.get(self.url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            df.to_csv(
                f"{self.path_to_source}/{self.dataset_name.replace(' ', '_')}_{self.publish_datetime_from.strftime('%Y%m%d')}_{self.publish_datetime_to.strftime('%Y%m%d')}.csv",
                index=False,
            )
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
        else:
            print(f"Failed to download data: {response.status_code}")


class WINDFORLatestDownloader(ELEXONDownloader):
    """
    Historic view of the latest forecasted wind generation (WINDFOR)
    This endpoint provides the latest wind generation forecast data. This provides wind generation forecast for wind farms which are
    visible to the ESO and have operational metering. Updated data is published by NGESO up to 8 times a day at 03:30, 05:30, 08:30, 10:30, 12:30, 16:30, 19:30 and 23:30.
    Results are filtered by a range of DateTime parameters.
    """

    def __init__(self, start_date, end_date):
        super().__init__()
        self.dataset_name = "WINDFOR_Latest"
        self.data = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.endpoint = "forecast/generation/wind/latest"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            "from": self.start_date.strftime("%Y-%m-%d"),
            "to": self.end_date.strftime("%Y-%m-%d"),
            "format": "json",
        }

        url = self.build_url(self.endpoint)
        try:
            data = self.make_request(url, params)
            df = pd.DataFrame(data)
            # Process data as needed
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
            # Optionally, upload to Azure Blob Storage
        except Exception as e:
            print(f"Error downloading {self.dataset_name}: {e}")


class WINDFORDownloader(ELEXONDownloader):
    """
    Wind generation forecast (WINDFOR)
    This endpoint provides wind generation forecast data.
    Specific publish time filters may be supplied, otherwise this will retrieve the latest published forecast.
    """

    def __init__(self, publish_datetime_from, publish_datetime_to):
        super().__init__()
        self.dataset_name = "WINDFOR"
        self.data = pd.DataFrame()
        self.publish_datetime_from = publish_datetime_from
        self.publish_datetime_to = publish_datetime_to
        self.endpoint = "datasets/WINDFOR"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            "publishDateTimeFrom": self.publish_datetime_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "publishDateTimeTo": self.publish_datetime_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json",
        }

        url = self.build_url(self.endpoint)
        try:
            data = self.make_request(url, params)
            df = pd.DataFrame(data)
            # Process data as needed
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.publish_datetime_from.strftime('%Y%m%d%H%M')}_{self.publish_datetime_to.strftime('%Y%m%d%H%M')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
            # Optionally, upload to Azure Blob Storage
        except Exception as e:
            print(f"Error downloading {self.dataset_name}: {e}")


class NDFLatestDownloader(ELEXONDownloader):
    """
    Historic view of the latest forecasted demand (NDF, TSDF)
    This endpoint allows for retrieving latest day-ahead demand forecast data from National Grid ESO.
    Results are filtered by settlement time, and only the latest published forecast for each settlement period is shown.
    """

    def __init__(self, start_date, end_date):
        super().__init__()
        self.dataset_name = "NDF_Latest"
        self.data = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.endpoint = "forecast/demand/day-ahead/latest"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            "from": self.start_date.strftime("%Y-%m-%d"),
            "to": self.end_date.strftime("%Y-%m-%d"),
            "format": "json",
        }
        url = self.build_url(self.endpoint)
        try:
            data = self.make_request(url, params)
            df = pd.DataFrame(data)
            # Process data as needed
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
            # Optionally, upload to Azure Blob Storage
        except Exception as e:
            print(f"Error downloading {self.dataset_name}: {e}")


class NDFDownloader(ELEXONDownloader):
    """
    Day and day-ahead National Demand forecast (NDF)
    This endpoint provides the National Demand forecast received from NGESO. Data is available daily and will show values for the day ahead.
    Expressed as an average MW value for each settlement period. The forecast is based on historically metered generation output for Great Britain.
    This value INCLUDES transmission losses, but EXCLUDES interconnector flows and demand from station transformers and pumped storage units.
    This API endpoint provides a maximum data output range of 1 day.
    Specific publish time filters may be supplied, otherwise this will retrieve the latest published forecast.
    """

    def __init__(self, publish_datetime_from, publish_datetime_to):
        super().__init__()
        self.dataset_name = "NDF"
        self.data = pd.DataFrame()
        self.publish_datetime_from = publish_datetime_from
        self.publish_datetime_to = publish_datetime_to
        self.endpoint = "datasets/NDF"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            "publishDateTimeFrom": self.publish_datetime_from.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "publishDateTimeTo": self.publish_datetime_to.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "format": "json",
        }

        url = self.build_url(self.endpoint)
        try:
            data = self.make_request(url, params)
            df = pd.DataFrame(data)
            # Process data as needed
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.publish_datetime_from.strftime('%Y%m%d%H%M')}_{self.publish_datetime_to.strftime('%Y%m%d%H%M')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
            # Optionally, upload to Azure Blob Storage
        except Exception as e:
            print(f"Error downloading {self.dataset_name}: {e}")


class TEMPDownloader(ELEXONDownloader):
    """
    This endpoint provides daily average GB temperature data (in Celsius) as well as reference temperatures (low, normal and high).
    This average data is calculated by National Grid ESO from the data retrieved from 6 weather stations around Britain.
    NGESO use this data as part of the electricity demand forecasting process.
    Date parameters must be provided in the exact format yyyy-MM-dd.
    """

    def __init__(self, from_date, to_date):
        super().__init__()
        self.dataset_name = "TEMP"
        self.data = pd.DataFrame()
        self.from_date = from_date  # datetime.date object
        self.to_date = to_date  # datetime.date object
        self.endpoint = "temperature"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            "from": self.from_date.strftime("%Y-%m-%d"),
            "to": self.to_date.strftime("%Y-%m-%d"),
            "format": "json",
        }

        url = self.build_url(self.endpoint)
        try:
            data = self.make_request(url, params)
            df = pd.DataFrame(data)
            # Process data as needed
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.from_date.strftime('%Y%m%d')}_{self.to_date.strftime('%Y%m%d')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
            # Optionally, upload to Azure Blob Storage
        except Exception as e:
            print(f"Error downloading {self.dataset_name}: {e}")


class DGWSDownloader(ELEXONDownloader):
    """
    Day-ahead generation forecast for wind and solar (DGWS/B1440)

    This endpoint provides day-ahead forecast generation data for wind and solar.
    Maximum data output range is 7 days.
    Filters available for settlement periods and process types.

    Parameters:
    -----------
    start_date : datetime
        Start date for the data request
    end_date : datetime
        End date for the data request
    settlement_period_from : int, optional
        Start settlement period (1-50), defaults to 1
    settlement_period_to : int, optional
        End settlement period (1-50), defaults to 50
    process_type : str, optional
        Type of process. Can be 'day ahead', 'intraday process', 'intraday total', or 'all'
    """

    def __init__(self, start_date, end_date, settlement_period_from=1, settlement_period_to=50, process_type='all'):
        super().__init__()
        self.dataset_name = "DGWS"
        self.data = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.settlement_period_from = settlement_period_from
        self.settlement_period_to = settlement_period_to
        self.process_type = process_type
        self.endpoint = "forecast/generation/wind-and-solar/day-ahead"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            'from': self.start_date.strftime("%Y-%m-%d"),
            'to': self.end_date.strftime("%Y-%m-%d"),
            'settlementPeriodFrom': self.settlement_period_from,
            'settlementPeriodTo': self.settlement_period_to,
            'processType': self.process_type,
            'format': 'json',
        }

        url = self.build_url(self.endpoint)
        try:
            data = self.make_request(url, params)
            if isinstance(data, dict) and 'data' in data:
                data = data['data']

            df = pd.DataFrame(data)

            # Convert timestamp columns to datetime
            timestamp_columns = ['publishTime', 'startTime']
            for col in timestamp_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            # Save to CSV
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")

        except Exception as e:
            print(f"Error downloading {self.dataset_name}: {e}")


class DISEBSPDownloader(ELEXONDownloader):
    """
    Downloader for Settlement System Prices (DISEBSP).

    Returns settlement system buy and sell prices generated by the SAA for a given settlement period, relating to the data for a settlement run.
    Only messages generated for the latest settlement run are returned.
    Settlement date parameter must be provided in the exact format yyyy-MM-dd.
    """

    def __init__(self, start_date, end_date):
        super().__init__()
        self.dataset_name = "DISEBSP"
        self.data = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.endpoint = "balancing/settlement/system-prices"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        data_list = []
        date_range = pd.date_range(self.start_date, self.end_date)
        settlement_periods = range(1, 51)  # Settlement periods from 1 to 50 inclusive
        for date in date_range:
            date_str = date.strftime("%Y-%m-%d")
            for period in settlement_periods:
                url = f"{self.build_url(self.endpoint)}/{date_str}/{period}"
                params = {
                    "format": "json",
                }
                try:
                    data = self.make_request(url, params)
                    if data:
                        # If data is a list, extend; if it's a dict, append
                        if isinstance(data, list):
                            data_list.extend(data)
                        else:
                            data_list.append(data)
                except Exception as e:
                    print(f"Error downloading data for {date_str} period {period}: {e}")
        if data_list:
            df = pd.DataFrame(data_list)
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
            # Optionally, upload to Azure Blob Storage
        else:
            print(f"No data found for {self.dataset_name} between {self.start_date} and {self.end_date}")


class FUELHHDownloader(ELEXONDownloader):
    """
    This endpoint provides the half-hourly generation outturn (Generation By Fuel type) to give our users an indication of the
    generation outturn for Great Britain. The data is aggregated by Fuel Type category and updated at 30-minute intervals
    with average MW values over 30 minutes for each category.
    This endpoint includes additional settlement parameters such as Settlement Date and Settlement Period.
    The Settlement Date fields cannot be set when a Publish Date field is set.
    Settlement date parameters must be provided in the exact format yyyy-MM-dd.
    """

    def __init__(self, start_date, end_date):
        super().__init__()
        self.dataset_name = "FUELHH"
        self.data = pd.DataFrame()
        self.start_date = start_date
        self.end_date = end_date
        self.endpoint = "datasets/FUELHH"

    def get_dataset_name(self):
        return self.dataset_name

    def updata_blob_data(self):
        params = {
            "settlementDateFrom": self.start_date.strftime("%Y-%m-%d"),
            "settlementDateTo": self.end_date.strftime("%Y-%m-%d"),
            "format": "json",
        }

        url = self.build_url(self.endpoint)
        try:
            data = self.make_request(url, params)
            df = pd.DataFrame(data)
            # Process data as needed
            file_name = f"{self.path_to_source}/{self.dataset_name}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
            df.to_csv(file_name, index=False)
            self.data = df
            print(f"Data for {self.dataset_name} downloaded successfully.")
            # Optionally, upload to Azure Blob Storage
        except Exception as e:
            print(f"Error downloading {self.dataset_name}: {e}")


class ELEXONDownloaderManager:
    """
    ELEXONDownloaderManager is a class that manages the download of ELEXON data.
    """

    def __init__(self, downloader_factory=None):
        if downloader_factory is None:
            self.downloader_factory = ELEXONDownloaderFactory()
        else:
            self.downloader_factory = downloader_factory

    def download(self, dataset_name, *args, **kwargs):
        downloader = self.downloader_factory.create_downloader(dataset_name, *args, **kwargs)
        downloader.updata_blob_data()
        return downloader.data


class ELEXONDownloaderFactory:
    """
    ELEXONDownloaderFactory is a class that creates ELEXON downloaders based on
    the dataset name provided.
    """

    def __init__(self):
        self.downloaders = {
            'ATL': ATLDownloader,
            'DATL': DATLDownloader,
            'WINDFOR_Latest': WINDFORLatestDownloader,
            'WINDFOR': WINDFORDownloader,
            'NDF_Latest': NDFLatestDownloader,
            'NDF': NDFDownloader,
            'TEMP': TEMPDownloader,
            'DISEBSP': DISEBSPDownloader,
            'FUELHH': FUELHHDownloader,
            'DGWS': DGWSDownloader,
        }

    def create_downloader(self, dataset_name, *args, **kwargs):
        downloader_class = self.downloaders.get(dataset_name)
        if downloader_class is None:
            raise ValueError(f"No downloader registered for dataset: {dataset_name}")
        return downloader_class(*args, **kwargs)
