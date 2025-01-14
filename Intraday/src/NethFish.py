# Execute the cURL command and capture the output

import numpy as np
import pandas as pd
from src.BotTrader import BotTrader, TradeValidator


class NethFish:
    def __init__(self, config):
        self.url = config["URL"]
        self.apifrom_key = config["API_KEY_FROM"]
        self.apito_key = config["API_KEY_TO"]
        self.delivery_area = config["DELIVERY_AREA"]
        self.delivery_from = config["DELIVERY_START_FROM"]
        self.delivery_to = config["DELIVERY_START_TO"]
        self.portfoliofrom_id = config["PORTFOLIO_ID_FROM"]
        self.portfolioto_id = config["PORTFOLIO_ID_TO"]
        self.settings = config["settings"]
        self.powerbotfrom = BotTrader(self.url, self.apifrom_key, self.delivery_area)
        self.powerbotto = BotTrader(self.url, self.apito_key, self.delivery_area)
        self.trades_from_opener = None  #
        self.trades_from_closer = None
        self.trades = None
        self.system_sell = None
        self.joined_trades_system_sell = None
        self.first_trade_to_imbalance = None

    def get_trades(self):

        # Dowload position opener's data
        open_data = self.powerbotfrom.download_trade_data(
            self.delivery_from, self.delivery_to, self.delivery_area, self.portfoliofrom_id
        )
        TradeValidator.validate_dataframe(open_data)
        self.trades_from_opener = open_data

        # Dowload position closer's data
        close_data = self.powerbotto.download_trade_data(
            self.delivery_from, self.delivery_to, self.delivery_area, self.portfolioto_id
        )
        TradeValidator.validate_dataframe(close_data)
        self.trades_from_closer = close_data

        # Join the trades from the opener and closer
        self.trades = pd.concat([self.trades_from_opener, self.trades_from_closer], ignore_index=True)

    def get_weighted_average_prices(self) -> pd.DataFrame:
        """
        Compute average buy and sell prices for each contract and algorithm (opener or closer).
        """

        opener_data = pd.DataFrame()
        closer_data = pd.DataFrame()

        for side_filter in ['buy', 'sell']:
            for algorith in ['opener', 'closer']:

                if algorith == 'opener':
                    algo_filter = self.trades_from_opener
                else:
                    algo_filter = self.trades_from_closer

                algo_filter = algo_filter[algo_filter[side_filter] == 1]
                weighted_mean = algo_filter.groupby(['contract_id', side_filter]).apply(
                    lambda x: np.average(x['price'], weights=x['quantity']), include_groups=False
                )

                # Reset index to make contract_id and buy accessible for merging
                weighted_mean = weighted_mean.reset_index()

                # Rename the calculated column
                weighted_mean = weighted_mean.rename(columns={0: 'weighted_mean_price'})

                # Merge the weighted means back to the original dataframe
                algo_filter = algo_filter.merge(weighted_mean, on=['contract_id', side_filter], how='left')

                if algorith == 'opener':
                    opener_data = pd.concat([opener_data, algo_filter])
                else:
                    closer_data = pd.concat([closer_data, algo_filter])

        self.trades_from_opener = opener_data
        self.trades_from_closer = closer_data

        return opener_data

    def get_weighted_average_minutes_to_delivery_start(self) -> pd.DataFrame:
        """
        Compute average minutes to delivery for each contract and algorithm (opener or closer).
        """

        opener_data = pd.DataFrame()
        closer_data = pd.DataFrame()

        for algorith in ['opener', 'closer']:

            if algorith == 'opener':
                algo_filter = self.trades_from_opener
            else:
                algo_filter = self.trades_from_closer

            # Calculate the difference in hours between the delivery start and the execution time
            algo_filter.loc[:, 'diff_hour'] = (
                (algo_filter.loc[:, 'delivery_start'] - algo_filter.loc[:, 'exec_time']).dt.total_seconds() / 60 / 60
            )
            weighted_mean = algo_filter.groupby('contract_id').apply(
                lambda x: np.average(x['diff_hour'], weights=x['quantity']), include_groups=False
            )

            # Reset index to make contract_id and buy accessible for merging
            weighted_mean = weighted_mean.reset_index()

            # Rename the calculated column
            weighted_mean = weighted_mean.rename(columns={0: 'weighted_minutes_to_delivery_start'})

            # Merge the weighted means back to the original dataframe
            algo_filter = algo_filter.merge(weighted_mean, on=['contract_id'], how='left')

            if algorith == 'opener':
                opener_data = pd.concat([opener_data, algo_filter])
            else:
                closer_data = pd.concat([closer_data, algo_filter])

        self.trades_from_opener = opener_data
        self.trades_from_closer = closer_data

        return opener_data

    def get_profit_loss(self) -> pd.DataFrame:
        """
        Compute profit and loss for each trade.
        """

        trades_boosted = pd.concat([self.trades_from_opener, self.trades_from_closer], ignore_index=True)
        mean_opener_time = pd.DataFrame(
            self.trades_from_opener.groupby('contract_id')['weighted_minutes_to_delivery_start'].mean()
        ).rename(columns={'weighted_minutes_to_delivery_start': 'weighted_minutes_to_delivery_start_opener'})
        mean_closer_time = pd.DataFrame(
            self.trades_from_closer.groupby('contract_id')['weighted_minutes_to_delivery_start'].mean()
        ).rename(columns={'weighted_minutes_to_delivery_start': 'weighted_minutes_to_delivery_start_closer'})
        mean_times = pd.merge(mean_opener_time, mean_closer_time, on='contract_id', how='outer')

        # by contract_id group trades and then use the weighted_mean_price for sell to compute how much was received for sold MW, then use weighted_mean_price with buy to compute how much was paid for bought MW
        # pseudo code
        #    df groupby('contract_id')
        #       for each 'buy' true, get the sum of quantity * weighted_mean_price
        #       for each 'sell' true, get the sum of quantity * weighted_mean_price
        #       profit_loss = subtract money paid (bought) from money received (sold)
        # return profit and loss for each contract_id as a dataframe

        # trades_boosted['profit_loss'] = .groupby('contract_id') trades_boosted['quantity'] * (trades_boosted['weighted_mean_price'] - trades_boosted['buy'])

        # Calculate money for each trade
        bought_quantity = pd.DataFrame(trades_boosted.groupby(['contract_id', 'buy'])['quantity'].sum())
        sold_quantity = pd.DataFrame(trades_boosted.groupby(['contract_id', 'sell'])['quantity'].sum())
        bought_weighted_mean = pd.DataFrame(
            trades_boosted.groupby(['contract_id', 'buy']).apply(
                lambda x: np.average(x['price'], weights=x['quantity']), include_groups=False
            )
        )
        sold_weighted_mean = pd.DataFrame(
            trades_boosted.groupby(['contract_id', 'sell']).apply(
                lambda x: np.average(x['price'], weights=x['quantity']), include_groups=False
            )
        )

        bought_value = pd.merge(bought_quantity, bought_weighted_mean, on='contract_id', how='outer').rename(
            columns={'quantity': 'quantity_bought', 0: 'price_bought'}
        )

        sold_value = pd.merge(sold_quantity, sold_weighted_mean, on='contract_id', how='outer').rename(
            columns={'quantity': 'quantity_sold', 0: 'price_sold'}
        )

        # combine the bought and sold values
        trades_means = pd.merge(bought_value, sold_value, on='contract_id', how='outer')
        trades_means['profit'] = (
            trades_means['quantity_sold'] * trades_means['price_sold']
            - trades_means['quantity_bought'] * trades_means['price_bought']
        )

        trades_means['pnl'] = trades_means['profit'].cumsum()
        trades_means['winning'] = trades_means['profit'] > 0

        trades_means_times = pd.merge(trades_means, mean_times, on='contract_id', how='left')

        final = pd.merge(trades_boosted, trades_means_times, on='contract_id', how='left')

        return final

    def scatter_plot_for_hours(
        self,
        df,
        x_variable,
        y_variable,
        boolean_variable,
        size_variable,
        title="Scatter Plot",
        x_label_str=None,
        y_label_str=None,
        size_scale=20,
    ):  # scale factor for point sizes
        """
        Create a scatter plot with customized point colors and sizes.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing all required variables
        x_variable : str
            Name of the column to plot on x-axis
        y_variable : str
            Name of the column to plot on y-axis
        boolean_variable : str
            Name of the column containing boolean values for color coding
        size_variable : str
            Name of the column containing float values for point sizes
        title : str, optional
            Plot title (default: "Scatter Plot")
        size_scale : float, optional
            Scaling factor for point sizes (default: 100)

        Returns:
        --------
        fig, ax : tuple
            Matplotlib figure and axis objects
        """
        import matplotlib.pyplot as plt

        if x_label_str is None:
            x_label_str = x_variable

        if y_label_str is None:
            y_label_str = y_variable

        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create scatter plot
        scatter = ax.scatter(
            x=df[x_variable],
            y=df[y_variable],
            c=df[boolean_variable],  # color based on boolean
            s=abs(df[size_variable]) * size_scale,  # size based on float variable
            alpha=0.6,  # slight transparency
            cmap='coolwarm',
        )  # color map for boolean values

        # Customize plot
        ax.set_xlabel(x_label_str)
        ax.set_ylabel(y_label_str)
        ax.set_title(title)

        # Add legend
        legend1 = ax.legend(*scatter.legend_elements(), title=boolean_variable)
        ax.add_artist(legend1)

        # Add size legend
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4, func=lambda s: s / size_scale)
        legend2 = ax.legend(handles, labels, title=size_variable, loc="upper left")

        plt.tight_layout()
        return fig, ax


# # get sum of trades
# bought_amount = contracts_to_analyze.groupby(['contract_id', 'buy'])['quantity'].sum()
# sold_amount = contracts_to_analyze.groupby(['contract_id', 'sell'])['quantity'].sum()
# buy_price_avg = contracts_to_analyze.groupby(['contract_id', 'buy'])['price'].mean()
# sell_price_avg = contracts_to_analyze.groupby(['contract_id', 'sell'])['price'].mean()

# # compute profit and loss bought amount times buy price - sold amount times sell price
# bought_amount = pd.DataFrame(bought_amount)
# sold_amount = pd.DataFrame(sold_amount)
# buy_price_avg = pd.DataFrame(buy_price_avg)
# sell_price_avg = pd.DataFrame(sell_price_avg)
# buy_sell = pd.merge(bought_amount, sold_amount, on='contract_id', how='outer')
# buy_sell.rename(columns={'quantity_x': 'quantity_bought', 'quantity_y': 'quantity_sold'}, inplace=True)
# buy_sell = pd.merge(buy_sell, buy_price_avg, on='contract_id', how='outer')
# buy_sell.rename(columns={'price': 'price_bought'}, inplace=True)
# buy_sell = pd.merge(buy_sell, sell_price_avg, on='contract_id', how='outer')
# buy_sell.rename(columns={'price': 'price_sold'}, inplace=True)
# buy_sell = buy_sell.fillna(0)
# buy_sell['profit'] = buy_sell['quantity_sold'] * buy_sell['price_sold'] - buy_sell['quantity_bought'] * buy_sell['price_bought']
# buy_sell['pnl'] = buy_sell['profit'].cumsum()

# with_hours = pd.merge(buy_sell, avg_hours, on='contract_id', how='left')
# with_hours = pd.merge(with_hours, dates_of_trades, on='contract_id', how='left')
