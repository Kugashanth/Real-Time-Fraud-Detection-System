from sklearn.calibration import CalibratedClassifierCV


def calibrate_classifier_isotonic(fitted_estimator, X_val, y_val) -> CalibratedClassifierCV:
    """Calibrates raw classifier probabilities on the validation split using Isotonic Regression."""
    calibrated = CalibratedClassifierCV(estimator=fitted_estimator, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated
