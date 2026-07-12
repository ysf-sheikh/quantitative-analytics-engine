import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_fico_rating_map(df, num_buckets=5):
    """
    Creates FICO rating buckets by maximizing log-likelihood
    of observed defaults using dynamic programming.
    """

    fico_min = 300
    fico_max = 850

    # Aggregate borrowers by FICO score
    score_data = (
        df.groupby("fico_score")["default"]
        .agg(["count", "sum"])
        .reindex(range(fico_min, fico_max + 1), fill_value=0)
    )

    counts = score_data["count"].values
    defaults = score_data["sum"].values

    score_range = len(counts)


    # Calculate log-likelihood for every possible bucket range
    bucket_cost = np.full(
        (score_range, score_range),
        -np.inf
    )

    for start in range(score_range):
        total_count = 0
        total_defaults = 0

        for end in range(start, score_range):

            total_count += counts[end]
            total_defaults += defaults[end]

            if total_count > 0:

                probability_default = (
                    total_defaults / total_count
                )

                # Avoid log(0) errors
                probability_default = np.clip(
                    probability_default,
                    1e-10,
                    1 - 1e-10
                )

                bucket_cost[start, end] = (
                    total_defaults * np.log(probability_default)
                    +
                    (total_count - total_defaults)
                    * np.log(1 - probability_default)
                )


    # Dynamic programming tables
    dp = np.full(
        (num_buckets + 1, score_range),
        -np.inf
    )

    split_points = np.zeros(
        (num_buckets + 1, score_range),
        dtype=int
    )


    # Base case: one bucket covering scores from start to end
    for end in range(score_range):
        dp[1][end] = bucket_cost[0][end]


    # Build optimal bucket combinations
    for bucket in range(2, num_buckets + 1):

        for end in range(bucket - 1, score_range):

            for split in range(bucket - 2, end):

                value = (
                    dp[bucket - 1][split]
                    +
                    bucket_cost[split + 1][end]
                )

                if value > dp[bucket][end]:
                    dp[bucket][end] = value
                    split_points[bucket][end] = split


    # Recover optimal boundaries
    boundaries = []

    current = score_range - 1

    for bucket in range(num_buckets, 1, -1):

        current = split_points[bucket][current]

        boundaries.append(current)


    boundaries = sorted(boundaries)

    fico_boundaries = [
        fico_min + boundary
        for boundary in boundaries
    ]


    # Create bucket summary
    bucket_summary = []

    for i in range(num_buckets):

        lower = (
            fico_min
            if i == 0
            else fico_boundaries[i - 1] + 1
        )

        upper = (
            fico_max
            if i == num_buckets - 1
            else fico_boundaries[i]
        )


        # Avoid overlapping ranges
        if i == num_buckets - 1:
            bucket_data = df[
                (df["fico_score"] >= lower)
                &
                (df["fico_score"] <= upper)
            ]
        else:
            bucket_data = df[
                (df["fico_score"] >= lower)
                &
                (df["fico_score"] < upper)
            ]


        borrowers = len(bucket_data)

        defaults_in_bucket = (
            bucket_data["default"].sum()
        )

        pd_rate = (
            defaults_in_bucket / borrowers
            if borrowers > 0
            else 0
        )


        bucket_summary.append(
            {
                "FICO Range": f"{lower}-{upper}",
                "Borrowers": borrowers,
                "Defaults": defaults_in_bucket,
                "PD": pd_rate
            }
        )


    # Sort buckets so lower rating means lower default risk
    bucket_summary = sorted(
        bucket_summary,
        key=lambda x: x["PD"]
    )


    # Assign final ratings
    for rating, bucket in enumerate(bucket_summary, 1):
        bucket["Rating"] = rating


    rating_map = pd.DataFrame(bucket_summary)

    return rating_map



# Load mortgage borrower data
df = pd.read_csv("loan_data.csv")


# Create FICO rating categories
rating_map = create_fico_rating_map(
    df,
    num_buckets=5
)


print("\nFICO Rating Map")
print("----------------")
print(
    rating_map[
        [
            "Rating",
            "FICO Range",
            "Borrowers",
            "Defaults",
            "PD"
        ]
    ].to_string(index=False)
)


# Visualize default probability by rating bucket
plt.figure(figsize=(8, 5))

plt.bar(
    rating_map["Rating"].astype(str),
    rating_map["PD"]
)

plt.xlabel("FICO Rating (1 = Lowest Risk)")
plt.ylabel("Probability of Default")
plt.title("Mortgage Default Risk by FICO Rating Bucket")

plt.show()
