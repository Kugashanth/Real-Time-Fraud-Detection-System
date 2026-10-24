from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE


def create_resampling_pipeline(
    classifier,
    sampling_strategy: float = 0.15,
    method: str = "smoteenn",
    random_state: int = 42,
) -> ImbPipeline:
    """
    Wraps RobustScaler + SMOTE variant + Classifier inside an imblearn Pipeline.
    Guarantees SMOTE is applied strictly inside training folds during cross-validation,
    leaving validation and test sets completely untouched (zero data leakage).
    """
    method_lower = method.lower()
    if method_lower == "smoteenn":
        sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    elif method_lower == "smotetomek":
        sampler = SMOTETomek(sampling_strategy=sampling_strategy, random_state=random_state)
    else:
        sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

    return ImbPipeline(
        steps=[
            ("scaler", RobustScaler()),
            ("resampler", sampler),
            ("classifier", classifier),
        ]
    )
