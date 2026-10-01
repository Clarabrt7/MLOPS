import wandb
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance

def main():
    run = wandb.init(project="MLOps-Lab2_MLOPS")
    config = wandb.config

    df = pd.read_csv("features.csv").dropna()
    
    X = df.drop(columns=["lap_duration"]) 
    y = df["lap_duration"]
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=42,
    )
    
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    wandb.log({
        "train_r2": train_score,
        "test_r2": test_score,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
    })

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    importance_table = wandb.Table(
        data=list(zip(X.columns, perm.importances_mean, perm.importances_std)),
        columns=["feature", "importance_mean", "importance_std"],
    )
    wandb.log({"feature_importance": importance_table})

    dataset_artifact = wandb.Artifact("f1-features-dataset", type="dataset")
    dataset_artifact.add_file("features.csv")
    run.log_artifact(dataset_artifact)

    joblib.dump(model, "model.joblib")
    model_artifact = wandb.Artifact(f"model-{run.id}", type="model", metadata=dict(config))
    model_artifact.add_file("model.joblib")
    run.log_artifact(model_artifact)

if __name__ == "__main__":
    main()