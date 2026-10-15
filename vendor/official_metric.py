"""Unchanged offline class definitions extracted from the archived official metric.
Server orchestration and Hugging Face authentication/logging code are omitted.
See docs/PROVENANCE.md. Original names containing micro_f2 are retained for compatibility.
"""
import abc, math, warnings
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, fbeta_score

class BaseMetric(abc.ABCMeta):
    """
    This class only depends on DataFrame and is independent from Hugging Face Hub variables.

    Class variables
    - COL_NAMES: canonical column names of the submission file
    - PRIMARY_ID_COL_NAME: column name of the index used to split public/private samples
    - ID_COL_NAMES: column name(s) to merge & sort the dataframes specified in sort order
    - LABEL_COL_NAMES: column name(s) of prediction labels
    - FILLNA: fillna value to use if submission is truncated
    - METRIC_NAMES: metric name conversion table from system output to leaderboard display
    """

    COL_NAMES: list[str] = None
    PRIMARY_ID_COL_NAME: str = None
    ID_COL_NAMES: list[str] = None
    LABEL_COL_NAMES: list[str] = None
    FILLNA = None

    # [{"system" : string, "display" : string}, ...]
    METRIC_NAMES: list[dict[Literal["system", "display"], str]] = None

    @classmethod
    def compute(
        cls,
        solution_df: pd.DataFrame,
        submission_df: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:

        _submission_df = cls.preprocess_submission(solution_df, submission_df)

        if cls._has_split_column(solution_df):
            # For shared task organizers
            result = cls._compute_metric_on_hf_space(solution_df, _submission_df)
        else:
            # For participants
            result = cls._compute_metric_locally(solution_df, _submission_df)
        return result

    @classmethod
    def preprocess_submission(
        cls, solution_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> pd.DataFrame:
        return cls._preprocess_merge(solution_df, cls._preprocess_clean(submission_df))

    @classmethod
    def _preprocess_merge(
        cls, solution_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> pd.DataFrame:
        merged_df = pd.merge(
            solution_df,
            submission_df,
            on=cls.ID_COL_NAMES,
            how="left",
            suffixes=("_solution", "_submission"),
        ).fillna(cls.FILLNA)

        submission_cols = []

        for col in cls.COL_NAMES:
            if col in cls.ID_COL_NAMES:
                submission_cols.append(col)
            elif col in submission_df.columns:
                # To avoid errors when some columns are missing
                submission_cols.append(col + "_submission")

        new_submission_df = merged_df.loc[:, submission_cols]

        submission_col_name_map = {
            name: name.replace("_submission", "") for name in submission_cols
        }

        new_submission_df = new_submission_df.rename(columns=submission_col_name_map)
        return new_submission_df

    @classmethod
    def _preprocess_clean(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return cls._sort_by_index(
            cls._normalize_label_format(cls._normalize_column_names(df))
        )

    @classmethod
    def _sort_by_index(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(cls.ID_COL_NAMES)

    @classmethod
    def _normalize_column_names(cls, submission_df: pd.DataFrame) -> pd.DataFrame:
        submission_dict = submission_df.to_dict()
        new_submission_dict = {}

        for key in submission_dict.keys():
            new_submission_dict[cls._normalize_column_name(key)] = submission_dict[key]

        return pd.DataFrame(new_submission_dict)

    @classmethod
    def _normalize_column_name(cls, column_name: str) -> str:
        if column_name in cls.COL_NAMES:
            # already canonical; no normalization needed
            return column_name
        else:
            variants = [
                column_name.upper(),
                column_name.lower(),
                column_name.title(),
                column_name.replace("-", "_"),
                column_name.replace("_", "-"),
                column_name.upper().replace("-", "_"),
                column_name.lower().replace("-", "_"),
                column_name.title().replace("-", "_"),
                column_name.upper().replace("_", "-"),
                column_name.lower().replace("_", "-"),
                column_name.title().replace("_", "-"),
            ]
            for variant in variants:
                if variant in cls.COL_NAMES:
                    # normalize column name
                    return variant
            # failed to normalize and give up
            warnings.warn(
                f"Unknown column name detected: {column_name}. Scores may not be calculated."
            )
            return column_name

    @staticmethod
    @abc.abstractmethod
    def _normalize_label_format(submission_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    @abc.abstractmethod
    def _compute(
        soludion_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> dict[str, float]:
        pass

    @staticmethod
    def _has_split_column(solution_df: pd.DataFrame) -> bool:
        return "split" in solution_df.columns

    @classmethod
    def _compute_metric_on_hf_space(
        cls,
        solution_df: pd.DataFrame,
        submission_df: pd.DataFrame,
    ) -> dict:
        (
            public_solution_df,
            public_submission_df,
            private_solution_df,
            private_submission_df,
        ) = cls.public_private_split(solution_df, submission_df)

        public_solution_df = public_solution_df.drop("split", axis=1)
        private_solution_df = private_solution_df.drop("split", axis=1)

        # NOTE: result must have two keys: "public_score" and "private_score"
        # NOTE: "public_score" and "private_score" must have the same keys
        result: dict[str, dict[str, float]] = {}

        if len(public_solution_df) * len(public_submission_df) > 0:
            result["public_score"] = cls._to_display_metric_name(
                cls._compute(public_solution_df, public_submission_df)
            )
        else:
            result["public_score"] = {
                metric_name_mapping["display"]: 0.0
                for metric_name_mapping in cls.METRIC_NAMES
            }

        if len(private_solution_df) * len(private_submission_df) > 0:
            result["private_score"] = cls._to_display_metric_name(
                cls._compute(private_solution_df, private_submission_df)
            )
        else:
            result["private_score"] = {
                metric_name_mapping["display"]: 0.0
                for metric_name_mapping in cls.METRIC_NAMES
            }

        # Convert np.nan to 0.0 to avoid error on Hugging Face Hub Space
        result = cls._nan_to_zero(result)

        return result

    @classmethod
    def _compute_metric_locally(
        cls, solution_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> dict:
        result: dict[str, dict[str, float]] = {}

        if len(solution_df) * len(submission_df) > 0:
            result["score"] = cls._compute(solution_df, submission_df)
        else:
            result["score"] = {
                metric_name_mapping["display"]: 0.0
                for metric_name_mapping in cls.METRIC_NAMES
            }

        result = cls._nan_to_zero(result)
        return result

    @classmethod
    def _to_display_metric_name(cls, metric_dict: dict[str, float]) -> dict[str, float]:
        new_dict = {}
        for name_map in cls.METRIC_NAMES:
            new_dict[name_map["display"]] = metric_dict[name_map["system"]]
        return new_dict

    @staticmethod
    def _nan_to_zero(
        score_dict: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        for partition in score_dict.keys():
            for name in score_dict[partition].keys():
                if math.isnan(score_dict[partition][name]):
                    score_dict[partition][name] = 0.0
        return score_dict

    @classmethod
    def public_private_split(
        cls, solution_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> tuple[pd.DataFrame]:
        # https://huggingface.co/docs/competitions/en/custom_metric
        public_ids = solution_df[solution_df.split == "public"][
            cls.PRIMARY_ID_COL_NAME
        ].values
        private_ids = solution_df[solution_df.split == "private"][
            cls.PRIMARY_ID_COL_NAME
        ].values

        public_solution_df = (
            solution_df[solution_df[cls.PRIMARY_ID_COL_NAME].isin(public_ids)]
            .sort_values(cls.PRIMARY_ID_COL_NAME)
            .reset_index(drop=True)
        )
        public_submission_df = (
            submission_df[submission_df[cls.PRIMARY_ID_COL_NAME].isin(public_ids)]
            .sort_values(cls.PRIMARY_ID_COL_NAME)
            .reset_index(drop=True)
        )

        print("Public/private split & sort by ID on submission file ...")
        print(f"Public LB submission: {public_submission_df.head()}")

        private_solution_df = (
            solution_df[solution_df[cls.PRIMARY_ID_COL_NAME].isin(private_ids)]
            .sort_values(cls.PRIMARY_ID_COL_NAME)
            .reset_index(drop=True)
        )
        private_submission_df = (
            submission_df[submission_df[cls.PRIMARY_ID_COL_NAME].isin(private_ids)]
            .sort_values(cls.PRIMARY_ID_COL_NAME)
            .reset_index(drop=True)
        )
        print(f"Private LB submission: {private_submission_df.head()}")

        return (
            public_solution_df,
            public_submission_df,
            private_solution_df,
            private_submission_df,
        )

class MainTaskMetric(BaseMetric):
    COL_NAMES = ["id", "t", "n", "m"]
    PRIMARY_ID_COL_NAME = "id"
    ID_COL_NAMES = ["id"]
    LABEL_COL_NAMES = ["t", "n", "m"]
    FILLNA = ""
    METRIC_NAMES = [
        {"system": "joint_accuracy_fine", "display": "Joint accuracy (fine)"},
        {"system": "t_accuracy_fine", "display": "T accuracy (fine)"},
        {"system": "n_accuracy_fine", "display": "N accuracy (fine)"},
        {"system": "m_accuracy_fine", "display": "M accuracy (fine)"},
        {"system": "joint_accuracy_coarse", "display": "Joint accuracy (coarse)"},
        {"system": "t_accuracy_coarse", "display": "T accuracy (coarse)"},
        {"system": "n_accuracy_coarse", "display": "N accuracy (coarse)"},
        {"system": "m_accuracy_coarse", "display": "M accuracy (coarse)"},
    ]

    @classmethod
    def _normalize_label_format(cls, submission_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolves label format variance. For example, for column "t":
            T2A -> T2a
            t2a -> T2a
            t2A -> T2a
            2a -> T2a
        """
        # e.g., {"t": {0: "T2a", 1: "T2b", ...}, "n" : {...} ...}
        submission_dict = submission_df.to_dict()

        for category in cls.LABEL_COL_NAMES:
            if category in submission_dict.keys():
                for i in range(len(submission_df)):
                    # Caution: "T2a".title() returns "T2A"
                    # T2A, t2a, t2A, 2a -> t2a, t2a, t2a, 2a
                    submission_dict[category][i] = (
                        str(submission_dict[category][i]).strip().lower()
                    )
                    if submission_dict[category][i].startswith(category):
                        # t2a -> T2a
                        submission_dict[category][i] = (
                            category.upper() + submission_dict[category][i][1:]
                        )
                    else:
                        # 2a -> T2a
                        submission_dict[category][i] = (
                            category.upper() + submission_dict[category][i]
                        )

        return pd.DataFrame(submission_dict)

    @classmethod
    def _get_coarse_labels(cls, label_df: pd.DataFrame) -> pd.DataFrame:
        """
        Input
        -----
        label_df: solution/submission dataframe that has columns ["id", "t", "n", "m"] at least
        """
        # e.g., {"t": {0: "T2a", 1: "T2b", ...}, "n" : {...} ...}
        label_dict = label_df.to_dict()

        for category in cls.LABEL_COL_NAMES:
            if category in label_df.columns:
                for i in range(len(label_df)):
                    label_dict[category][i] = cls._to_coarse_label(
                        category, label_dict[category][i]
                    )

        return pd.DataFrame(label_dict)

    @classmethod
    def _to_coarse_label(cls, category: str, fine_label: str) -> str:
        """
        Convert fine label to coarse label.
        Non-canonical label will convert to empty string.
        (e.g., "T2a" -> "T2", "T2" -> "")
        """
        mapping = {
            "t": {
                "T0": "T0",
                "Tis": "T1",
                "T1mi": "T1",
                "T1a": "T1",
                "T1b": "T1",
                "T1c": "T1",
                "T2a": "T2",
                "T2b": "T2",
                "T3": "T3",
                "T4": "T4",
            },
            "n": {"N0": "N0", "N1": "N1", "N2": "N2", "N3": "N3"},
            "m": {"M0": "M0", "M1a": "M1", "M1b": "M1", "M1c": "M1"},
        }

        try:
            return mapping[category][fine_label]
        except KeyError:
            return cls.FILLNA

    @classmethod
    def _compute(
        cls, solution_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> dict[str, float]:

        result = {}
        sample_wise_result = {}

        categories = list(set(cls.LABEL_COL_NAMES) & set(submission_df.columns))
        missing_categories = list(
            set(cls.LABEL_COL_NAMES) - set(submission_df.columns)
        )  # if submission lacks any column

        # Fine score
        for category in categories:
            # pd.Series containing np.bool_
            sample_wise_result[category] = (
                solution_df[category] == submission_df[category]
            )

        if missing_categories:
            result["joint_accuracy_fine"] = 0.0
        else:
            result["joint_accuracy_fine"] = np.all(
                list(sample_wise_result.values()), axis=0
            ).mean()

        for category in categories:
            result[category + "_accuracy_fine"] = accuracy_score(
                solution_df[category], submission_df[category]
            )

        for category in missing_categories:
            result[category + "_accuracy_fine"] = 0.0

        # Coarse score
        coarse_solution_df = cls._get_coarse_labels(solution_df)
        coarse_submission_df = cls._get_coarse_labels(submission_df)

        coarse_sample_wise_result = {}

        for category in categories:
            # pd.Series containing np.bool_
            coarse_sample_wise_result[category] = (
                coarse_solution_df[category] == coarse_submission_df[category]
            )

        if missing_categories:
            result["joint_accuracy_coarse"] = 0.0
        else:
            result["joint_accuracy_coarse"] = np.all(
                list(coarse_sample_wise_result.values()), axis=0
            ).mean()

        for category in categories:
            result[category + "_accuracy_coarse"] = accuracy_score(
                coarse_solution_df[category], coarse_submission_df[category]
            )

        for category in missing_categories:
            result[category + "_accuracy_coarse"] = 0.0

        return result

class SubTaskMetric(BaseMetric):
    COL_NAMES = [
        "id",
        "sentence_index",
        "omittable",
        "measure",
        "extension",
        "atelectasis",
        "satellite",
        "lymphadenopathy",
        "pleural",
        "distant",
    ]
    PRIMARY_ID_COL_NAME = "id"
    ID_COL_NAMES = ["id", "sentence_index"]
    LABEL_COL_NAMES = [
        "omittable",
        "measure",
        "extension",
        "atelectasis",
        "satellite",
        "lymphadenopathy",
        "pleural",
        "distant",
    ]
    FILLNA = math.nan
    METRIC_NAMES = [
        {"system": "overall_micro_f2", "display": "Overall micro F2.0"},
        {"system": "inclusion_micro_f2", "display": "Inclusion micro F2.0"},
        {"system": "measure_micro_f2", "display": "Measure micro F2.0"},
        {"system": "extension_micro_f2", "display": "Extension micro F2.0"},
        {"system": "atelectasis_micro_f2", "display": "Atelectasis micro F2.0"},
        {"system": "satellite_micro_f2", "display": "Satellite micro F2.0"},
        {"system": "lymphadenopathy_micro_f2", "display": "Lymphadenopathy micro F2.0"},
        {"system": "pleural_micro_f2", "display": "Pleural micro F2.0"},
        {"system": "distant_micro_f2", "display": "Distant micro F2.0"},
    ]

    @classmethod
    def _normalize_label_format(cls, submission_df: pd.DataFrame) -> pd.DataFrame:
        # e.g., {"omittable": {0: 1, 1: 1, ...}, ...}
        submission_dict = submission_df.to_dict()

        for category in cls.LABEL_COL_NAMES:
            if category in submission_dict.keys():
                for i in range(len(submission_df)):
                    submission_dict[category][i] = int(
                        str(submission_dict[category][i]).strip()
                    )

        return pd.DataFrame(submission_dict)

    @classmethod
    def _compute(
        cls, solution_df: pd.DataFrame, submission_df: pd.DataFrame
    ) -> dict[str, float]:
        result = {}

        missing_categories = list(
            set(cls.LABEL_COL_NAMES) - set(submission_df.columns)
        )  # if submission lacks any column

        missing_sample_indices = submission_df.isna().any(
            axis=1
        )  # if submission lacks any record
        missing_sample_ratio = missing_sample_indices.sum() / len(solution_df)

        # Multi-column metrics
        if missing_categories and (missing_categories != ["omittable"]):
            result["overall_micro_f2"] = 0.0
        else:
            sample_wise_result = []
            for i in range(len(solution_df)):
                if missing_sample_indices.iloc[i]:
                    sample_wise_result.append(0.0)
                else:
                    sample_wise_result.append(
                        fbeta_score(
                            solution_df.iloc[i, :].loc["measure":"distant"],
                            submission_df.iloc[i, :].loc["measure":"distant"],
                            beta=2.0,
                            pos_label=1,
                            average="binary",
                            zero_division=1.0,  # Give score 1.0 for correct prediction on all-negative records
                        )
                    )
            print(sample_wise_result)
            result["overall_micro_f2"] = np.mean(sample_wise_result)

        # Single-column metrics
        if "omittable" in missing_categories:
            result["inclusion_micro_f2"] = 0.0
        else:
            result["inclusion_micro_f2"] = fbeta_score(
                solution_df["omittable"][~missing_sample_indices],
                submission_df["omittable"][~missing_sample_indices],
                beta=2.0,
                pos_label=0,
                average="binary",
                zero_division=1.0,
            ) * (1 - missing_sample_ratio)

        for colname in cls.LABEL_COL_NAMES[1:]:
            if colname in missing_categories:
                result[f"{colname}_micro_f2"] = 0.0
            else:
                result[f"{colname}_micro_f2"] = fbeta_score(
                    solution_df[colname][~missing_sample_indices],
                    submission_df[colname][~missing_sample_indices],
                    beta=2.0,
                    pos_label=1,
                    average="binary",
                    zero_division=1.0,
                ) * (1 - missing_sample_ratio)

        return result
