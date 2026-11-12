"""Dataset registry for the bundled tabular benchmarks."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder: str
    files: tuple
    format: str = "csv"
    target_columns: tuple = field(default_factory=tuple)
    id_columns: tuple = field(default_factory=tuple)
    categorical_columns: tuple = field(default_factory=tuple)
    description: str = ""


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "online_news": DatasetSpec(
        "online_news", "online_news_popularity", ("OnlineNewsPopularity.csv",),
        target_columns=("shares",), id_columns=("url",),
        description="Online News Popularity; shares is retained as the target, URL is an identifier.",
    ),
    "adult": DatasetSpec(
        "adult", "adult", ("adult.data", "adult.test"), "adult",
        target_columns=("income",),
        categorical_columns=("workclass", "education", "marital_status", "occupation",
                             "relationship", "race", "sex", "native_country"),
        description="UCI Adult/Census Income; train and test rows are combined for unsupervised imputation.",
    ),
    "credit_default": DatasetSpec(
        "credit_default", "default_credit_card_clients", ("default_of_credit_card_clients.xlsx",
        "default of credit card clients.xls"), "credit",
        target_columns=("default payment next month",), id_columns=("ID",),
        categorical_columns=("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3",
                             "PAY_4", "PAY_5", "PAY_6"),
        description="Default of Credit Card Clients; ID and default-payment outcome are excluded from predictors.",
    ),
    "online_shoppers": DatasetSpec(
        "online_shoppers", "online_shoppers", ("online_shoppers_intention.csv",),
        target_columns=("Revenue",),
        categorical_columns=("Month", "OperatingSystems", "Browser", "Region", "TrafficType",
                             "VisitorType", "Weekend"),
        description="Online Shoppers Purchasing Intention; Revenue is retained as target.",
    ),
    "letter_recognition": DatasetSpec(
        "letter_recognition", "letter_recognition", ("letter-recognition.data",), "letter",
        target_columns=("letter",),
        description="Letter Recognition; the letter class is retained as target and 16 attributes are predictors.",
    ),
    "bike_daily": DatasetSpec(
        "bike_daily", "bike_sharing", ("day.csv",), "bike",
        target_columns=("cnt", "casual", "registered"), id_columns=("instant", "dteday"),
        categorical_columns=("season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"),
        description="Bike Sharing daily records; rental-count outcomes and row/date identifiers are excluded.",
    ),
    "bike_hourly": DatasetSpec(
        "bike_hourly", "bike_sharing", ("hour.csv",), "bike",
        target_columns=("cnt", "casual", "registered"), id_columns=("instant", "dteday"),
        categorical_columns=("season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"),
        description="Bike Sharing hourly records; rental-count outcomes and row/date identifiers are excluded.",
    ),
    "obesity": DatasetSpec(
        "obesity", "obesity_levels", ("ObesityDataSet_raw_and_data_sinthetic.csv",),
        target_columns=("NObeyesdad",),
        categorical_columns=("Gender", "family_history_with_overweight", "FAVC", "CAEC",
                             "SMOKE", "SCC", "CALC", "MTRANS"),
        description="Estimation of Obesity Levels; NObeyesdad is retained as target.",
    ),
    "wine_red": DatasetSpec(
        "wine_red", "wine_quality", ("winequality-red.csv",), "semicolon",
        target_columns=("quality",), description="Wine Quality red variant; quality is retained as target.",
    ),
    "wine_white": DatasetSpec(
        "wine_white", "wine_quality", ("winequality-white.csv",), "semicolon",
        target_columns=("quality",), description="Wine Quality white variant; quality is retained as target.",
    ),
    "diabetes": DatasetSpec(
        "diabetes", "", (), "sklearn", target_columns=("target",),
        description="Scikit-learn Diabetes regression dataset (10 input features; target retained separately).",
    ),
}

ALIASES = {
    "online_news_popularity": "online_news",
    "default_of_credit_card_clients": "credit_default",
    "default_credit_card_clients": "credit_default",
    "online_shoppers_intention": "online_shoppers",
    "letter": "letter_recognition",
    "bike": "bike_daily",
    "bike_sharing_daily": "bike_daily",
    "bike_sharing_hourly": "bike_hourly",
    "obesity_levels": "obesity",
    "wine_red_quality": "wine_red",
    "wine_white_quality": "wine_white",
}


def canonical_dataset_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = ALIASES.get(key, key)
    if key not in DATASET_SPECS:
        choices = ", ".join(DATASET_SPECS)
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {choices}")
    return key
