import json

# Execute the cURL command and capture the output
import subprocess
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
from powerbot_client import (
    ApiClient,
    Configuration,
    TradesApi,
)
from pydantic import BaseModel, ValidationError, field_validator, model_validator


class BotTrader:

    def __init__(self, api_url, api_key, delivery_area):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {"accept": "application/json", "api_key": api_key}
        self.delivery_area = delivery_area

        # Initialize PowerBot client for internal trades
        self.client = ApiClient(Configuration(api_key={"api_key_security": self.api_key}, host=self.api_url))
        self.trades_api = TradesApi(self.client)

    def download_internal_trades(
        self,
        portfolio_ids=None,
        from_execution_time=None,
        to_execution_time=None,
        delivery_within_start=None,
        delivery_within_end=None,
        batch_size=500,
    ):
        """
        Download internal trades in daily chunks and convert to pandas DataFrame.

        Args:
            portfolio_ids (list): List of portfolio IDs to filter trades
            from_execution_time (datetime): Filter trades executed after this time
            to_execution_time (datetime): Filter trades executed before this time
            delivery_within_start (datetime): Filter trades with delivery starting after this
            delivery_within_end (datetime): Filter trades with delivery ending before this
            batch_size (int): Number of trades to fetch per request

        Returns:
            pd.DataFrame: DataFrame containing internal trades with properly formatted columns
        """
        try:
            all_trades = []

            # Process data in daily chunks
            current_date = from_execution_time
            while current_date < to_execution_time:
                # Calculate next date
                next_date = min(current_date + timedelta(days=1), to_execution_time)

                print(
                    f"Downloading data for {current_date.strftime('%Y-%m-%dT%H:%M:%SZ')} to {next_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                )

                # Initialize pagination for this day's chunk
                offset = 0

                while True:
                    # Fetch batch of trades for current day
                    trades = self.trades_api.get_internal_trades(
                        portfolio_id=portfolio_ids,
                        from_execution_time=current_date,
                        to_execution_time=next_date,
                        delivery_within_start=delivery_within_start,
                        delivery_within_end=delivery_within_end,
                        offset=offset,
                        limit=batch_size,
                    )

                    # Break if no more trades for this day
                    if not trades:
                        break

                    # Process each trade JSON into a flat dictionary
                    for trade in trades:
                        trade_dict = {
                            # Trade identification
                            'internal_trade_id': trade.internal_trade_id,
                            'exchange': trade.exchange,
                            'exec_time': trade.exec_time,
                            'api_timestamp': trade.api_timestamp,
                            # Contract information
                            'contract_id': trade.contract_id,
                            'contract_name': trade.contract_name,
                            'delivery_start': trade.delivery_start,
                            'delivery_end': trade.delivery_end,
                            'prod': trade.prod,
                            # Trade details
                            'price': trade.price,
                            'quantity': trade.quantity,
                            # Buy side details
                            'buy_order_id': trade.buy_order_id,
                            'buy_cl_order_id': trade.buy_cl_order_id,
                            'buy_txt': trade.buy_txt,
                            'buy_aggressor_indicator': trade.buy_aggressor_indicator,
                            'buy_portfolio_id': trade.buy_portfolio_id,
                            'buy_delivery_area': trade.buy_delivery_area,
                            'buy_strategy_id': trade.buy_strategy_id,
                            # Sell side details
                            'sell_order_id': trade.sell_order_id,
                            'sell_cl_order_id': trade.sell_cl_order_id,
                            'sell_txt': trade.sell_txt,
                            'sell_aggressor_indicator': trade.sell_aggressor_indicator,
                            'sell_portfolio_id': trade.sell_portfolio_id,
                            'sell_delivery_area': trade.sell_delivery_area,
                            'sell_strategy_id': trade.sell_strategy_id,
                        }
                        all_trades.append(trade_dict)

                    # Update offset for next batch
                    offset += batch_size

                    # Break if less than batch size returned (end of day's data reached)
                    if len(trades) < batch_size:
                        break

                # Move to next day
                current_date = next_date

            # Convert to DataFrame
            df = pd.DataFrame(all_trades)

            # Convert timestamp columns to datetime
            timestamp_cols = ['exec_time', 'api_timestamp', 'delivery_start', 'delivery_end']
            for col in timestamp_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)

            # Sort by execution time
            if 'exec_time' in df.columns:
                df = df.sort_values('exec_time')

            # Set index to internal_trade_id but keep it as a column
            if 'internal_trade_id' in df.columns:
                df.set_index('internal_trade_id', drop=False, inplace=True)

            return df

        except Exception as e:
            print(f"Error downloading internal trades: {str(e)}")
            return pd.DataFrame()

    def download_data(self, delivery_from, delivery_to, contract_duration_minutes, set_params):

        # Function to format datetime objects
        def format_datetime(dt):
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Convert delivery_from and delivery_to to datetime objects
        delivery_from_dt = datetime.strptime(delivery_from, "%Y-%m-%dT%H:%M:%SZ")
        delivery_to_dt = datetime.strptime(delivery_to, "%Y-%m-%dT%H:%M:%SZ")

        # Generate list of date ranges with a maximum span of 48 hours
        def generate_time_ranges(start_date, end_date, max_hours=48):
            ranges = []
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + timedelta(hours=max_hours), end_date)
                ranges.append((current_start, current_end))
                current_start = current_end
            return ranges

        time_ranges = generate_time_ranges(delivery_from_dt, delivery_to_dt, max_hours=48)

        # Initialize a list to store DataFrames for each parameter set
        dataframes = []

        counter = 0
        for idx, params_set in enumerate(set_params):

            print(f"Processing parameter set {idx + 1}")
            # Initialize a list to store data for the current parameter set
            data_list = []

            # Loop over each time range
            for start_date, end_date in time_ranges:
                counter += 1
                # at every 10 days wait 1 seconds
                if counter % 10 == 0:
                    counter = 0
                    print("Waiting for 1 seconds")
                    time.sleep(1)

                # Define the base parameters
                params = {
                    "contract_duration_minutes": contract_duration_minutes,
                    "delivery_area": self.delivery_area,
                    "delivery_from": format_datetime(start_date),
                    "delivery_to": format_datetime(end_date),
                    "includeHistoricData": "True",
                    "async_req": "True",
                }

                # Update the parameters with the set-specific values, excluding None values
                for key, value in params_set.items():
                    if value is not None:
                        params[key] = value

                # Build the query string
                query_string = urllib.parse.urlencode(params)

                # Construct the full URL
                full_url = f"{self.api_url}?{query_string}"

                # Build the cURL command
                curl_command = f'curl -X GET "{full_url}" \\\n'

                for key, value in self.headers.items():
                    curl_command += f'    -H "{key}: {value}" \\\n'

                # Remove the last backslash and newline
                curl_command = curl_command.rstrip(' \\\n')

                # Display the cURL command
                print(f"Constructed cURL Command on day: {format_datetime(start_date)}")
                print(curl_command)

                result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)

                # Check for errors
                if result.returncode == 0:
                    print("Response received.")
                    # print(result.stdout)
                else:
                    print("Error:")
                    print(result.stderr)

                # Check for errors
                if result.returncode == 0:
                    # Parse the JSON response
                    try:
                        json_data = json.loads(result.stdout)
                        # Extract the 'statistics' data
                        statistics = json_data.get('statistics', [])
                        if statistics:
                            df = pd.DataFrame(statistics)
                            data_list.append(df)
                    except json.JSONDecodeError as e:
                        print(
                            f"JSON decode error for parameter set {idx+1}, time range {start_date} to {end_date}: {e}"
                        )
                else:
                    print(
                        f"Error executing cURL command for parameter set {idx+1}, time range {start_date} to {end_date}:"
                    )
                    print(result.stderr)

            # Concatenate all data for the current parameter set
            if data_list:
                df_all = pd.concat(data_list, ignore_index=True)
                dataframes.append(df_all)
                print(f"Data collected for parameter set {idx + 1}: {len(df_all)} records")
            else:
                dataframes.append(pd.DataFrame())
                print(f"No data found for parameter set {idx + 1}")

        return dataframes

    def analyze_trades_by_contract(self, df):
        """
        Analyze trades by contract_id, calculating P&L and remaining positions.

        Args:
            df: DataFrame containing internal trades data

        Returns:
            DataFrame with aggregated trade analysis containing:
            - Total quantity traded
            - Net position (remaining quantity)
            - Average prices for buys/sells
            - Realized P&L
        """
        # Create trade direction column (-1 for buy, 1 for sell)
        df['trade_direction'] = 0
        df.loc[df['buy_txt'].notna(), 'trade_direction'] = -1  # Buy trades
        df.loc[df['sell_txt'].notna(), 'trade_direction'] = 1  # Sell trades

        # Calculate trade value (negative for buys, positive for sells)
        df['trade_value'] = df['trade_direction'] * df['quantity']
        df['cash_flow'] = -df['trade_direction'] * df['quantity'] * df['price']

        # Group by contract_id
        results = []
        for contract_id, contract_trades in df.groupby('contract_id'):
            buy_trades = contract_trades[contract_trades['trade_direction'] == -1]
            sell_trades = contract_trades[contract_trades['trade_direction'] == 1]

            # Calculate trade metrics
            total_bought = buy_trades['quantity'].sum() if not buy_trades.empty else 0
            total_sold = sell_trades['quantity'].sum() if not sell_trades.empty else 0

            # Calculate average prices
            avg_buy_price = (
                (buy_trades['price'] * buy_trades['quantity']).sum() / total_bought if total_bought > 0 else 0
            )
            avg_sell_price = (
                (sell_trades['price'] * sell_trades['quantity']).sum() / total_sold if total_sold > 0 else 0
            )

            # Calculate net position and P&L
            net_position = total_bought - total_sold
            total_cash_flow = contract_trades['cash_flow'].sum()

            # Calculate realized P&L
            realized_quantity = min(total_bought, total_sold)
            if realized_quantity > 0:
                realized_pl = (avg_sell_price - avg_buy_price) * realized_quantity
            else:
                realized_pl = 0

            # Get delivery period info
            delivery_start = contract_trades['delivery_start'].iloc[0]
            delivery_end = contract_trades['delivery_end'].iloc[0]

            results.append(
                {
                    'contract_id': contract_id,
                    'delivery_start': delivery_start,
                    'delivery_end': delivery_end,
                    'total_bought': total_bought,
                    'total_sold': total_sold,
                    'avg_buy_price': avg_buy_price,
                    'avg_sell_price': avg_sell_price,
                    'net_position': net_position,
                    'realized_quantity': realized_quantity,
                    'realized_pl': realized_pl,
                    'total_cash_flow': total_cash_flow,
                    'remaining_position': 'Long' if net_position > 0 else 'Short' if net_position < 0 else 'Flat',
                    'remaining_quantity': abs(net_position),
                    'remaining_price': avg_buy_price if net_position > 0 else avg_sell_price if net_position < 0 else 0,
                }
            )

        return pd.DataFrame(results).set_index('contract_id')

    def summarize_trading_activity(self, df):
        """
        Provide a summary of trading activity and P&L.
        """
        analysis = self.analyze_trades_by_contract(df)

        print("=== Trading Activity Summary ===")
        print(f"\nTotal Contracts Traded: {len(analysis)}")
        print(f"Total Realized P&L: {analysis['realized_pl'].sum():.2f}")
        print(f"Total Cash Flow: {analysis['total_cash_flow'].sum():.2f}")

        print("\nOpen Positions:")
        open_positions = analysis[analysis['net_position'] != 0]
        if not open_positions.empty:
            for _, pos in open_positions.iterrows():
                print(f"\nContract: {pos.name}")
                print(f"Delivery: {pos['delivery_start']} to {pos['delivery_end']}")
                print(f"Position: {pos['remaining_position']}")
                print(f"Quantity: {abs(pos['net_position']):.2f}")
                print(f"Average Price: {pos['remaining_price']:.2f}")
        else:
            print("No open positions")

        return analysis

    def download_trade_data(self, delivery_from, delivery_to, delivery_area: str, portfolio_id: str = None):
        """
        Download trades data from PowerBot in daily chunks between specified dates.

        Parameters:
        -----------
        delivery_from : str
            Start date in format "YYYY-MM-DDThh:mm:ssZ"
        delivery_to : str
            End date in format "YYYY-MM-DDThh:mm:ssZ"
        delivery_area : str
            The EIC code of the delivery area
        portfolio_id : str, optional
            The portfolio ID to filter for. If None, will get data for all accessible portfolios

        Returns:
        --------
        pd.DataFrame
            DataFrame containing trades data
        """

        # Function to format datetime objects
        def format_datetime(dt):
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # PowerBot requests to provide data in UTC, but internally converts that to two hours later, and then returns UTC
        # therefore we need to subtract two hours from the delivery_from and delivery_to dates
        delivery_from_dt = datetime.strptime(delivery_from, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        delivery_to_dt = datetime.strptime(delivery_to, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

        # Initialize empty list to store all trades
        all_trades_data = []

        # Process data in daily chunks
        current_date = delivery_from_dt
        while current_date < delivery_to_dt:
            # Calculate next date
            next_date = min(current_date + timedelta(days=1), delivery_to_dt)

            print(f"Downloading data for {format_datetime(current_date)} to {format_datetime(next_date)}")

            try:
                # Download trades for current chunk
                trades = self.trades_api.get_trades(
                    portfolio_id=[portfolio_id] if portfolio_id else None,
                    delivery_area=delivery_area,
                    delivery_within_start=current_date,
                    delivery_within_end=next_date,
                    limit=500,
                )

                # Process trades
                for trade in trades:
                    trade_row = {
                        'trade_id': trade.trade_id,
                        'state': trade.state,
                        'exchange': trade.exchange,
                        'delivery_area': trade.delivery_area,
                        'api_timestamp': trade.api_timestamp,
                        'exec_time': trade.exec_time,
                        'contract_id': trade.contract_id,
                        'contract_name': trade.contract_name,
                        'delivery_start': trade.delivery_start,
                        'delivery_end': trade.delivery_end,
                        'price': trade.price,
                        'quantity': trade.quantity,
                        'buy': trade.buy,
                        'sell': trade.sell,
                        'buy_portfolio_id': trade.buy_portfolio_id,
                        'sell_portfolio_id': trade.sell_portfolio_id,
                        'buy_txt': trade.buy_txt,
                        'sell_txt': trade.sell_txt,
                        'self_trade': trade.self_trade,
                    }
                    all_trades_data.append(trade_row)

            except Exception as e:
                print(f"Error downloading data for {format_datetime(current_date)}: {str(e)}")

            # Move to next day
            current_date = next_date

        # Convert to pandas DataFrame
        trades_df = pd.DataFrame(all_trades_data)

        # Sort DataFrame
        if not trades_df.empty:
            trades_df.sort_values('exec_time', inplace=True)

        return trades_df


class TradeRecord(BaseModel):
    """
    Pydantic model representing a single trade row.

    Required fields:
      - trade_id: str
      - api_timestamp: datetime (assumes timezone aware)
      - delivery_start: datetime (tz-aware)
      - delivery_end: datetime (tz-aware)
      - quantity: float (non-zero)
      - buy: bool
      - sell: bool

    Row-level checks:
      - quantity != 0
      - cannot have buy == sell == True
    """

    trade_id: str
    api_timestamp: datetime
    delivery_start: datetime
    delivery_end: datetime
    quantity: float
    buy: bool
    sell: bool

    @field_validator("quantity")
    def quantity_non_zero(cls, v):
        if abs(v) < 0.01:
            raise ValueError("quantity must be a non-zero float.")
        return v

    @model_validator(mode="after")
    def check_buy_sell_exclusivity(self):
        if self.buy and self.sell:
            raise ValueError("Cannot have both buy and sell be True within the same row.")
        return self


class TradeValidator:
    """
    Class for validating an entire DataFrame of trade data.

    Steps:
      1) Row-level validation with the TradeRecord Pydantic model.
      2) Ensure (trade_id, api_timestamp) is unique across rows.
    """

    def validate_dataframe(df: pd.DataFrame) -> None:
        """
        Validate that each row (trade_id, timestamps, etc.) is correct and
        check that (trade_id, api_timestamp) is unique across the entire dataset.

        Raises ValueError if any violation is found.
        """
        # -- 1. Row-level checks via Pydantic --
        trade_records = []
        for idx, row in df.iterrows():
            try:
                record = TradeRecord(
                    trade_id=str(row["trade_id"]),  # Ensure string
                    api_timestamp=row["api_timestamp"],
                    delivery_start=row["delivery_start"],
                    delivery_end=row["delivery_end"],
                    quantity=float(row["quantity"]),  # Ensure float
                    buy=bool(row["buy"]),
                    sell=bool(row["sell"]),
                )
                trade_records.append(record)
            except ValidationError as e:
                raise ValueError(f"Row {idx} failed validation: {e.errors()}")

        # -- 2. Cross-row checks --

        # 2.1 Check uniqueness of (trade_id, api_timestamp).
        duplicates = df.groupby(["trade_id", "api_timestamp"]).size().reset_index(name="count")
        dup_rows = duplicates[duplicates["count"] > 1]
        if not dup_rows.empty:
            # if you want to show all duplicates, you can:
            problem_pairs = dup_rows[["trade_id", "api_timestamp"]].values.tolist()
            raise ValueError(f"The following (trade_id, api_timestamp) pairs are not unique: {problem_pairs}")

        # If we get here, everything is valid. No return = success.
