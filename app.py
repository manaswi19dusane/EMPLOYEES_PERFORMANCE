from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
TRAIN_DATASET = BASE_DIR / "train_dataset.csv"
TEST_DATASET = BASE_DIR / "test_dataset.csv"
FEATURE_COLUMNS = ["over_time", "incentive", "idle_time", "no_of_workers", "smv"]
TARGET_COLUMN = "actual_productivity"


@st.cache_data
def load_data_from_sqlite() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_DATASET)
    test_df = pd.read_csv(TEST_DATASET)

    with sqlite3.connect(":memory:") as connection:
        train_df.to_sql("train_employees", connection, index=False, if_exists="replace")
        test_df.to_sql("test_employees", connection, index=False, if_exists="replace")

        train_query = """
            SELECT
                team,
                targeted_productivity,
                actual_productivity,
                smv,
                over_time,
                incentive,
                idle_time,
                no_of_workers,
                month
            FROM train_employees
        """
        test_query = """
            SELECT
                team,
                targeted_productivity,
                smv,
                over_time,
                incentive,
                idle_time,
                no_of_workers,
                month
            FROM test_employees
        """

        train_data = pd.read_sql_query(train_query, connection)
        test_data = pd.read_sql_query(test_query, connection)

    return train_data, test_data


@st.cache_data
def train_model(df: pd.DataFrame) -> tuple[LinearRegression, float, float]:
    clean_df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    x = clean_df[FEATURE_COLUMNS]
    y = clean_df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mse, r2


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    numeric_columns = FEATURE_COLUMNS + ["targeted_productivity", TARGET_COLUMN]
    correlation = df[numeric_columns].corr()

    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(correlation, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_yticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=35, ha="right")
    ax.set_yticklabels(correlation.columns)

    for row in range(len(correlation.index)):
        for col in range(len(correlation.columns)):
            value = correlation.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(
        page_title="Employee Performance Evaluation",
        page_icon=":bar_chart:",
        layout="wide",
    )

    train_df, test_df = load_data_from_sqlite()
    model, mse, r2 = train_model(train_df)

    actual_values = train_df[TARGET_COLUMN].to_numpy()
    average_productivity = np.mean(actual_values)
    productivity_std = np.std(actual_values)
    target_gap = np.mean(
        train_df[TARGET_COLUMN].to_numpy()
        - train_df["targeted_productivity"].to_numpy()
    )

    st.title("Employee Performance Evaluation")
    st.caption("Data is loaded from local CSV files through an in-memory SQLite database.")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Training Rows", f"{len(train_df):,}")
    metric_2.metric("Average Productivity", f"{average_productivity:.2f}")
    metric_3.metric("Productivity Std Dev", f"{productivity_std:.2f}")
    metric_4.metric("Actual vs Target Gap", f"{target_gap:+.2f}")

    st.subheader("Dataset Preview")
    st.dataframe(train_df.head(15), width="stretch")

    st.subheader("Team Productivity")
    team_productivity = (
        train_df.groupby("team")[["targeted_productivity", TARGET_COLUMN]]
        .mean()
        .sort_index()
    )
    st.bar_chart(team_productivity)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Correlation Heatmap")
        st.pyplot(plot_correlation_heatmap(train_df), width="stretch")

    with chart_right:
        st.subheader("Productivity Trend by Month")
        monthly_trend = (
            train_df.groupby("month")[["targeted_productivity", TARGET_COLUMN]]
            .mean()
            .sort_index()
        )
        st.line_chart(monthly_trend)

    st.subheader("Overtime vs Actual Productivity")
    scatter_data = train_df[["over_time", TARGET_COLUMN, "team"]].rename(
        columns={TARGET_COLUMN: "actual_productivity"}
    )
    st.scatter_chart(
        scatter_data,
        x="over_time",
        y="actual_productivity",
        color="team",
    )

    st.subheader("Model Performance")
    perf_1, perf_2 = st.columns(2)
    perf_1.metric("Mean Squared Error", f"{mse:.4f}")
    perf_2.metric("R2 Score", f"{r2:.4f}")

    st.subheader("Predict Actual Productivity")
    input_left, input_right = st.columns(2)

    with input_left:
        over_time = st.number_input("Overtime (minutes)", 0, 8000, 1000)
        incentive = st.number_input("Incentive", 0, 200, 50)
        idle_time = st.number_input("Idle Time", 0.0, 10.0, 0.0)

    with input_right:
        no_of_workers = st.number_input("Number of Workers", 1, 100, 30)
        smv = st.number_input("SMV", 0.0, 60.0, 10.0)

    if st.button("Predict"):
        input_values = np.array(
            [[over_time, incentive, idle_time, no_of_workers, smv]], dtype=float
        )
        prediction_df = pd.DataFrame(input_values, columns=FEATURE_COLUMNS)
        prediction = model.predict(prediction_df)[0]
        st.success(f"Predicted Actual Productivity: {prediction:.2f}")

    with st.expander("SQLite Source Tables"):
        st.write("Training data is queried from `train_employees`.")
        st.write("Test data is queried from `test_employees`.")
        st.dataframe(test_df.head(10), width="stretch")


if __name__ == "__main__":
    main()
