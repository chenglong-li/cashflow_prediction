# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np


class GetData():
    def __init__(self):
        file_path = os.path.join(os.getcwd(), 'dataset', 'user_balance_table.csv')
        self.user_balance = pd.read_csv(file_path)

    def get_day_balance(self):
        day_balance = self.user_balance[['report_date', 'yBalance', 'tBalance']]
        day_balance['report_date'] = pd.to_datetime(day_balance['report_date'], format=('%Y%m%d'))
        day_balance = day_balance.set_index('report_date')
        day_balance_sum = day_balance.resample('d').sum()

        return day_balance_sum

    def get_day_purchase(self):
        day_purchase = self.user_balance[['report_date', 'total_purchase_amt']]
        day_purchase['report_date'] = pd.to_datetime(day_purchase['report_date'], format=('%Y%m%d'))
        day_purchase = day_purchase.set_index('report_date')
        day_purchase = day_purchase.resample('d').sum()

        return day_purchase

    def get_day_redeem(self):
        day_redeem = self.user_balance[['report_date', 'total_redeem_amt']]
        day_redeem['report_date'] = pd.to_datetime(day_redeem['report_date'], format=('%Y%m%d'))
        day_redeem = day_redeem.set_index('report_date')
        day_redeem = day_redeem.resample('d').sum()

        return day_redeem


if __name__ == '__main__':
    data = GetData()
    print(np.shape(data.get_day_balance()))