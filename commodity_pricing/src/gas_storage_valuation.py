import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from interpolate_extrapolate import get_gas_price

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    volumes_injected,
    volumes_withdrawn,
    max_capacity,
    max_injection_rate,
    max_withdrawal_rate,
    storage_rate_per_day,
    injection_withdrawal_cost_per_unit
):
    """
    Calculates the value of a natural gas storage contract.
    """

    transactions = []

    # Create a list of all injections and withdrawals
    for date, volume in zip(injection_dates, volumes_injected):
        transactions.append({
            "date": pd.to_datetime(date),
            "type": "inject",
            "volume": volume
        })

    for date, volume in zip(withdrawal_dates, volumes_withdrawn):
        transactions.append({
            "date": pd.to_datetime(date),
            "type": "withdraw",
            "volume": volume
        })

    if not transactions:
        return 0.0

    transactions.sort(key=lambda x: x["date"])

    # Store transactions by date to allow multiple actions on the same day
    transaction_calendar = {}

    for tx in transactions:
        transaction_calendar.setdefault(tx["date"], []).append(tx)

    inventory = 0.0
    revenue = 0.0
    purchase_cost = 0.0
    operating_cost = 0.0

    start_date = transactions[0]["date"]
    end_date = transactions[-1]["date"]

    # Simulate inventory changes over the contract period
    for day in range((end_date - start_date).days + 1):

        current_date = start_date + pd.Timedelta(days=day)

        # Daily storage cost
        operating_cost += inventory * storage_rate_per_day

        if current_date in transaction_calendar:

            for tx in transaction_calendar[current_date]:

                price = get_gas_price(
                    current_date.strftime("%Y-%m-%d")
                )

                if tx["type"] == "inject":

                    volume = tx["volume"]

                    if volume > max_injection_rate:
                        raise ValueError(
                            f"Injection rate exceeded on {current_date.date()}"
                        )

                    if inventory + volume > max_capacity:
                        raise ValueError(
                            f"Storage capacity exceeded on {current_date.date()}"
                        )

                    inventory += volume
                    purchase_cost += volume * price
                    operating_cost += volume * injection_withdrawal_cost_per_unit

                elif tx["type"] == "withdraw":

                    volume = tx["volume"]

                    if volume > max_withdrawal_rate:
                        raise ValueError(
                            f"Withdrawal rate exceeded on {current_date.date()}"
                        )

                    if inventory - volume < 0:
                        raise ValueError(
                            f"Insufficient inventory on {current_date.date()}"
                        )

                    inventory -= volume
                    revenue += volume * price
                    operating_cost += volume * injection_withdrawal_cost_per_unit

    value = revenue - purchase_cost - operating_cost

    # Check whether all stored gas has been withdrawn
    if inventory != 0:
        print(f"Warning: {inventory} units remain in storage")

    return value
