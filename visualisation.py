import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path


def plot_heatmap_plotly(
        X,
        timestamps,
        animal_ids,
        out_dir,
        title="Heatmap",
        filename="heatmap.html",
        yaxis="Data",
        xaxis="Time",
):
    fig = make_subplots(rows=1, cols=1)
    trace = go.Heatmap(z=X, x=timestamps, y=animal_ids, colorscale="Viridis")
    fig.add_trace(trace, row=1, col=1)
    fig.update_layout(title_text=title)
    fig.update_layout(xaxis_title=xaxis)
    fig.update_layout(yaxis_title=yaxis)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename.replace("=", "_").lower()
    print(file_path)
    fig.write_html(str(file_path))
    fig.write_image(str(file_path).replace("html", "png"), format="png", scale=5)

    return trace, title


def main():
    # Set input directory
    input_dir = Path("data")
    output_dir = Path("output")
    files = list(input_dir.glob("*.csv"))

    # Define the filtering date
    start_date = datetime(2025, 1, 30)

    all_data = []

    for file_path in files:
        uid, date = file_path.stem.split('_')
        columns = ["timestamp", "x", "y", "z", "activity"]

        # Read data
        data = pd.read_csv(file_path, names=columns)
        data["datetime"] = data["timestamp"].astype(str).apply(lambda x: datetime.strptime(x, "%y%m%d%H%M%S"))

        # Filter data after the specified date
        data = data[data["datetime"] >= start_date]

        data = data.set_index("datetime")
        data_resampled = data.resample("1T").sum().reset_index() # 1T code resamples to 1min

        # Store UID in the data
        data_resampled["uid"] = uid
        all_data.append(data_resampled)

    # Combine all filtered data
    if all_data:
        df = pd.concat(all_data, ignore_index=True)

        # Sort data by datetime
        df = df.sort_values(by="datetime").reset_index(drop=True)

        # Create pivot table for heatmap
        heatmap_data = df.pivot(index="uid", columns="datetime", values="activity")

        # Extract required arrays
        X = np.sqrt(heatmap_data.values)  # Activity values (Z)
        timestamps = heatmap_data.columns
        timestamps = [pd.to_datetime(x) for x in timestamps]
        animal_ids = heatmap_data.index  # Y-axis labels (UIDs)

        # Call the provided heatmap function
        plot_heatmap_plotly(
            X,
            timestamps,
            animal_ids,
            output_dir,
            title="Cat Activity Heatmap (Filtered after 30/01/2025)",
            filename="cat_activity_heatmap.html",
            yaxis="Animal ID",
            xaxis="Time",
        )

        print("Heatmap saved successfully.")

        grouped = df.groupby(df["uid"])
        dfs = [group for _, group in grouped]

        for df in dfs:
            uid = df["uid"].values[0]
            fig_accel = px.line(
                df, x="datetime", y=["x", "y", "z"],
                labels={"datetime": "Time", "value": "Acceleration (m/s²)", "variable": "Axis"},
                title=f"{uid} Accelerometer Data Over Time (Resampled to 1 minute bins)"
            )
            fig_accel.write_html(output_dir / f"accelerometer_plot_{uid}.html")
            fig_accel.write_image(output_dir / f"accelerometer_plot_{uid}.png", format="png", scale=5)
            fig_activity = px.bar(
                df, x="datetime", y="activity",
                labels={"datetime": "Time", "activity": "Activity Count"},
                title=f"{uid} Activity Count Over Time (Resampled to 1 minute bins)"
            )
            fig_activity.write_html(output_dir / f"activity_plot_{uid}.html")
            fig_activity.write_image(output_dir / f"activity_plot_{uid}.png", format="png", scale=1)

    else:
        print("No data available after 30/01/2025.")


if __name__ == "__main__":
    main()