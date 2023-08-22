import requests
import datetime
import pandas as pd
import time


def fetch_binance_data(symbol='ETHUSDT', timeframe='1h', timerange=4320, max_retries=5,
                       wait_time=10):  # 4320 hours = 180 days
    def fetch_data_from_binance(start_time):
        retries = 0
        while retries < max_retries:
            try:
                endpoint = "https://fapi.binance.com/fapi/v1/klines"
                params = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'startTime': start_time
                }

                response = requests.get(endpoint, params=params)
                response.raise_for_status()  # Raise an error for bad responses
                return response.json()

            except requests.RequestException as e:
                print(f"Error fetching data: {e}. Retrying in {wait_time} seconds...")
                retries += 1
                time.sleep(wait_time)
        raise ValueError(f"Failed to fetch data after {max_retries} retries.")

    def adjust_dataframe(df):
        # Convert 'Open time' and 'Close time' to datetime format
        df['Open_time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close_time'] = pd.to_datetime(df['Close time'], unit='ms')

        # Convert 'Open_time' and 'Close_time' to string format
        df['Open_time'] = df['Open_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['Close_time'] = df['Close_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Sort dataframe by 'Open_time'
        df.sort_values(by='Open_time', inplace=True, ascending=True)

        # Identify time columns (based on prior naming convention)
        time_cols = [col for col in df.columns if "time" in col.lower()]

        # Convert all non-time columns to float
        for col in df.columns:
            if col not in time_cols:
                df[col] = df[col].astype(float)

        return df

    # Calculate timestamp for the desired time range ago
    time_range_ago = datetime.datetime.now() - datetime.timedelta(hours=timerange)
    timestamp_time_range_ago = int(time_range_ago.timestamp() * 1000)  # Convert to milliseconds

    # Initial data fetch
    data = fetch_data_from_binance(timestamp_time_range_ago)

    # Continuously fetch data until the current date is reached
    while True:
        last_timestamp = data[-1][0]
        current_timestamp = int(datetime.datetime.now().timestamp() * 1000)

        # Break if the last timestamp is within the last hour
        if current_timestamp - last_timestamp <= 60 * 60 * 1000:
            break

        new_data = fetch_data_from_binance(last_timestamp + 60 * 60 * 1000)
        data += new_data

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                     'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                     'Taker buy quote asset volume', 'Ignore'])

    # Adjust dataframe to match the desired format
    df = adjust_dataframe(df)

    return df


