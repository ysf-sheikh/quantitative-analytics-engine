import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


class FICORatingQuantizer:
    """
    Optimizes FICO score segmentation using Dynamic Programming.

    Features:
        - Maximum binomial log-likelihood segmentation
        - Minimum bucket population constraints
        - Monotonic PD enforcement
        - Bayesian PD smoothing
        - Weight of Evidence (WoE)
        - Information Value (IV)
        - Credit rating assignment
        - Visualization

    Notes:
        Monotonicity is enforced through iterative adjacent-bucket
        merging after unconstrained DP optimization. This does not
        guarantee globally optimal constrained segmentation.
    """

    def __init__(
        self,
        num_buckets=5,
        min_bucket_size=100,
        monotonic=True,
        smoothing=True,
        alpha=1,
        beta=1
    ):

        if not isinstance(num_buckets, int) or num_buckets < 1:
            raise ValueError("num_buckets must be a positive integer.")

        if not isinstance(min_bucket_size, int) or min_bucket_size < 1:
            raise ValueError("min_bucket_size must be a positive integer.")

        if alpha <= 0 or beta <= 0:
            raise ValueError("Alpha and beta must be positive.")

        self.num_buckets = num_buckets
        self.min_bucket_size = min_bucket_size
        self.monotonic = monotonic
        self.smoothing = smoothing
        self.alpha = alpha
        self.beta = beta

        self.rating_map = None
        self.boundaries = None
        self.is_fitted = False

    # ---------------------------------------------------------
    # 1. INPUT VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate_data(df):

        required_columns = {"fico_score", "default"}

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Dataset must contain columns: {required_columns}"
            )

        if df.empty:
            raise ValueError("Dataset cannot be empty.")

        data = df[["fico_score", "default"]].copy()

        if data.isnull().any().any():
            raise ValueError("Missing values detected.")

        if not pd.api.types.is_numeric_dtype(data["fico_score"]):
            raise ValueError("FICO scores must be numeric.")

        if not np.isfinite(data["fico_score"]).all():
            raise ValueError("FICO scores must be finite.")

        if not data["fico_score"].between(300, 850).all():
            raise ValueError("FICO scores must be between 300 and 850.")

        if not (data["fico_score"] % 1 == 0).all():
            raise ValueError("FICO scores must be integers.")

        if not data["default"].isin([0, 1]).all():
            raise ValueError("Default labels must contain only 0 and 1.")

        data["fico_score"] = data["fico_score"].astype(int)
        data["default"] = data["default"].astype(int)

        return data

    # ---------------------------------------------------------
    # 2. BINOMIAL LOG-LIKELIHOOD
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_log_likelihood(
        total_count,
        total_defaults
    ):

        if total_count <= 0:
            return -np.inf

        good = total_count - total_defaults

        likelihood = 0.0

        if total_defaults > 0:

            likelihood += (
                total_defaults
                * np.log(total_defaults / total_count)
            )

        if good > 0:

            likelihood += (
                good
                * np.log(good / total_count)
            )

        return likelihood

    # ---------------------------------------------------------
    # 3. DYNAMIC PROGRAMMING OPTIMIZATION
    # ---------------------------------------------------------

    def _optimize_boundaries(self, counts, defaults):

        n = len(counts)
        buckets = self.num_buckets

        if np.count_nonzero(counts) < buckets:
            raise ValueError(
                "Not enough distinct populated FICO scores "
                "for the requested number of buckets."
            )

        prefix_counts = np.concatenate([
            [0],
            np.cumsum(counts)
        ])

        prefix_defaults = np.concatenate([
            [0],
            np.cumsum(defaults)
        ])

        dp = np.full(
            (buckets + 1, n + 1),
            -np.inf
        )

        split_points = np.full(
            (buckets + 1, n + 1),
            -1,
            dtype=int
        )

        dp[0, 0] = 0.0

        for b in range(1, buckets + 1):

            for end in range(b, n + 1):

                for split in range(b - 1, end):

                    total_count = (
                        prefix_counts[end]
                        - prefix_counts[split]
                    )

                    if total_count < self.min_bucket_size:
                        continue

                    total_defaults = (
                        prefix_defaults[end]
                        - prefix_defaults[split]
                    )

                    bucket_likelihood = (
                        self._calculate_log_likelihood(
                            total_count,
                            total_defaults
                        )
                    )

                    previous_score = dp[b - 1, split]

                    if not np.isfinite(previous_score):
                        continue

                    candidate = (
                        previous_score
                        + bucket_likelihood
                    )

                    if candidate > dp[b, end]:

                        dp[b, end] = candidate

                        split_points[b, end] = split

        if not np.isfinite(dp[buckets, n]):
            raise ValueError(
                "Unable to create the requested number of buckets "
                "under the minimum bucket size constraint."
            )

        # Recover optimal segmentation.

        segments = []

        end = n

        for b in range(buckets, 0, -1):

            split = split_points[b, end]

            if split < 0:
                raise RuntimeError(
                    "Failed to reconstruct optimal boundaries."
                )

            segments.append((split, end))

            end = split

        segments.reverse()

        return segments

    # ---------------------------------------------------------
    # 4. MONOTONICITY ENFORCEMENT
    # ---------------------------------------------------------

    @staticmethod
    def _enforce_monotonicity(segments):

        """
        Merge adjacent buckets whenever PD increases with FICO.

        Expected relationship:

            PD(lower FICO) >= PD(higher FICO)

        This is an iterative merging heuristic, not a globally
        optimal monotonic segmentation algorithm.
        """

        segments = [segment.copy() for segment in segments]

        changed = True

        while changed:

            changed = False

            i = 0

            while i < len(segments) - 1:

                current = segments[i]
                following = segments[i + 1]

                current_pd = (
                    current["defaults"] / current["borrowers"]
                )

                following_pd = (
                    following["defaults"] / following["borrowers"]
                )

                if current_pd < following_pd:

                    merged = {
                        "start": current["start"],
                        "end": following["end"],
                        "borrowers": (
                            current["borrowers"]
                            + following["borrowers"]
                        ),
                        "defaults": (
                            current["defaults"]
                            + following["defaults"]
                        )
                    }

                    segments[i] = merged

                    segments.pop(i + 1)

                    changed = True

                    if i > 0:
                        i -= 1

                else:

                    i += 1

        return segments

    # ---------------------------------------------------------
    # 5. FIT MODEL
    # ---------------------------------------------------------

    def fit(self, df):

        data = self._validate_data(df)

        scores = np.sort(
            data["fico_score"].unique()
        )

        aggregated = (
            data.groupby("fico_score")["default"]
            .agg(["count", "sum"])
            .reindex(scores, fill_value=0)
        )

        counts = aggregated["count"].to_numpy(dtype=float)

        defaults = aggregated["sum"].to_numpy(dtype=float)

        # Dynamic Programming optimization.

        optimal_segments = self._optimize_boundaries(
            counts,
            defaults
        )

        segments = []

        for start, end in optimal_segments:

            segments.append({
                "start": int(start),
                "end": int(end),
                "borrowers": int(counts[start:end].sum()),
                "defaults": int(defaults[start:end].sum())
            })

        # Optional monotonic enforcement.

        if self.monotonic:

            segments = self._enforce_monotonicity(
                segments
            )

        self.segments_ = segments
        self.scores_ = scores

        self.global_pd_ = data["default"].mean()

        self.data_ = data

        self.rating_map = self._build_rating_map()

        self.boundaries = [
            segment["end"]
            for segment in segments[:-1]
        ]

        self.is_fitted = True

        return self

    # ---------------------------------------------------------
    # 6. BUILD RATING MAP
    # ---------------------------------------------------------

    def _build_rating_map(self):

        total_good = (
            len(self.data_)
            - self.data_["default"].sum()
        )

        total_bad = self.data_["default"].sum()

        rows = []

        for i, segment in enumerate(self.segments_):

            borrowers = segment["borrowers"]

            defaults = segment["defaults"]

            good = borrowers - defaults

            raw_pd = defaults / borrowers

            if self.smoothing:

                smoothed_pd = (
                    defaults + self.alpha
                ) / (
                    borrowers + self.alpha + self.beta
                )

            else:

                smoothed_pd = raw_pd

            good_distribution = (
                good / total_good
                if total_good > 0
                else 0
            )

            bad_distribution = (
                defaults / total_bad
                if total_bad > 0
                else 0
            )

            # Smoothed WoE calculations.

            epsilon = 1e-10

            smoothed_good = (
                good_distribution + epsilon
            )

            smoothed_bad = (
                bad_distribution + epsilon
            )

            woe = np.log(
                smoothed_good / smoothed_bad
            )

            iv_contribution = (
                smoothed_good - smoothed_bad
            ) * woe

            lower = (
                300
                if i == 0
                else self.scores_[segment["start"]]
            )

            upper = (
                850
                if i == len(self.segments_) - 1
                else self.scores_[segment["end"]]
            )

            rows.append({

                "FICO Lower": int(lower),

                "FICO Upper": int(upper),

                "FICO Range": f"{lower}-{upper}",

                "Borrowers": borrowers,

                "Defaults": defaults,

                "Good": good,

                "Raw PD": raw_pd,

                "PD": smoothed_pd,

                "WoE": woe,

                "IV Contribution": iv_contribution

            })

        rating_map = pd.DataFrame(rows)

        # Assign rating 1 to the lowest-risk bucket.

        rating_map = rating_map.sort_values(
            "PD",
            ascending=True
        ).reset_index(drop=True)

        rating_map["Rating"] = (
            np.arange(len(rating_map)) + 1
        )

        rating_map = rating_map[
            [
                "Rating",
                "FICO Lower",
                "FICO Upper",
                "FICO Range",
                "Borrowers",
                "Defaults",
                "Good",
                "Raw PD",
                "PD",
                "WoE",
                "IV Contribution"
            ]
        ]

        return rating_map

    # ---------------------------------------------------------
    # 7. PREDICT RATINGS
    # ---------------------------------------------------------

    def predict(self, fico_scores):

        if not self.is_fitted:
            raise RuntimeError("Fit the quantizer before prediction.")

        scores = np.asarray(fico_scores)

        if not np.issubdtype(scores.dtype, np.number):
            raise ValueError("FICO scores must be numeric.")

        if not np.isfinite(scores).all():
            raise ValueError("FICO scores must be finite.")

        if not np.equal(scores, np.floor(scores)).all():
            raise ValueError("FICO scores must be integers.")

        if not ((scores >= 300) & (scores <= 850)).all():
            raise ValueError("FICO scores must be between 300 and 850.")

        ratings = np.zeros(len(scores), dtype=int)

        for _, row in self.rating_map.iterrows():

            mask = (
                (scores >= row["FICO Lower"])
                &
                (scores <= row["FICO Upper"])
            )

            ratings[mask] = int(row["Rating"])

        return ratings

    # ---------------------------------------------------------
    # 8. INFORMATION VALUE
    # ---------------------------------------------------------

    def calculate_iv(self):

        if not self.is_fitted:
            raise RuntimeError("Fit the quantizer first.")

        return self.rating_map["IV Contribution"].sum()

    # ---------------------------------------------------------
    # 9. VISUALIZATION
    # ---------------------------------------------------------

    def plot_rating_map(self):

        if not self.is_fitted:
            raise RuntimeError("Fit the quantizer first.")

        rating_map = self.rating_map.sort_values(
            "FICO Lower"
        )

        plt.figure(figsize=(10, 6))

        plt.bar(
            rating_map["FICO Range"],
            rating_map["PD"]
        )

        plt.xlabel("FICO Score Range")

        plt.ylabel("Probability of Default")

        plt.title(
            "Optimized FICO Credit Risk Segmentation"
        )

        plt.xticks(rotation=30)

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        plt.show()

    # ---------------------------------------------------------
    # 10. SUMMARY
    # ---------------------------------------------------------

    def summary(self):

        if not self.is_fitted:
            raise RuntimeError("Fit the quantizer first.")

        print("\nFICO CREDIT RATING SUMMARY")

        print("=" * 75)

        print(
            self.rating_map.to_string(
                index=False,
                formatters={
                    "Raw PD": "{:.4%}".format,
                    "PD": "{:.4%}".format,
                    "WoE": "{:.4f}".format,
                    "IV Contribution": "{:.4f}".format
                }
            )
        )

        print("\nModel Statistics")

        print("-" * 35)

        print(f"Original requested buckets: {self.num_buckets}")

        print(f"Final buckets: {len(self.rating_map)}")

        print(f"Global default rate: {self.global_pd_:.4%}")

        print(f"Information Value: {self.calculate_iv():.4f}")

        print("=" * 75)

        return self.rating_map


# =============================================================
# BACKWARD-COMPATIBLE FUNCTION
# =============================================================

def create_fico_rating_map(
    df,
    num_buckets=5,
    min_bucket_size=100,
    monotonic=True,
    smoothing=True
):

    """
    Backward-compatible interface for FICO rating map creation.

    Returns:
        pandas.DataFrame: Optimized credit rating map.
    """

    quantizer = FICORatingQuantizer(

        num_buckets=num_buckets,

        min_bucket_size=min_bucket_size,

        monotonic=monotonic,

        smoothing=smoothing

    )

    quantizer.fit(df)

    return quantizer.rating_map


# =============================================================
# EXECUTION
# =============================================================

if __name__ == "__main__":

    data_path = Path(__file__).resolve().parent / "loan_data.csv"

    df = pd.read_csv(data_path)

    quantizer = FICORatingQuantizer(

        num_buckets=5,

        min_bucket_size=100,

        monotonic=True,

        smoothing=True

    )

    quantizer.fit(df)

    quantizer.summary()

    print("\nTotal Information Value:")

    print(round(quantizer.calculate_iv(), 4))

    quantizer.plot_rating_map()
