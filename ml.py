from pathlib import Path
import pandas as pd
import numpy as np
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go

from visualisation import plot_heatmap_plotly
import matplotlib.pyplot as plt


def anscombe(arr, sigma_sq=0, alpha=1):
    """
    Generalized Anscombe variance-stabilizing transformation
    References:
    [1] http://www.cs.tut.fi/~foi/invansc/
    [2] M. Makitalo and A. Foi, "Optimal inversion of the generalized
    Anscombe transformation for Poisson-Gaussian noise", IEEE Trans.
    Image Process, 2012
    [3] J.L. Starck, F. Murtagh, and A. Bijaoui, Image  Processing
    and Data Analysis, Cambridge University Press, Cambridge, 1998)
    :param arr: variance-stabilized signal
    :param sigma_sq: variance of the Gaussian noise component
    :param alpha: scaling factor of the Poisson noise component
    :return: variance-stabilized array
    """
    v = np.maximum((arr / alpha) + (3.0 / 8.0) + sigma_sq / (alpha**2), 0)
    f = 2.0 * np.sqrt(v)
    return f


def main(w=10):
    df = pd.read_csv('dataset.csv')
    print(df)

    uid_column = df.columns[0]
    time_columns = df.columns[1:]

    samples = []

    for _, row in df.iterrows():
        uid = row[uid_column]
        values = row.values[1:]
        num_windows = len(values) // w

        row_samples = [
            {
                "uid": uid,
                "start_date": time_columns[i * w],
                "data": values[i * w: (i + 1) * w]
            }
            for i in range(num_windows)
        ]

        samples.extend(row_samples)
    print(samples)
    clean_samples =[]
    for s in samples:
        d = s["data"]
        series = pd.Series(d)
        count_positive = (series > 0).sum()
        if count_positive < len(series) / 2:
            continue
        clean_samples.append(s)
    print(f"Found {len(clean_samples)} samples")
    df = pd.DataFrame(clean_samples)
    data_expanded = pd.DataFrame(df["data"].tolist())
    df_expanded = pd.concat([df.drop(columns=["data"]), data_expanded], axis=1)
    df_expanded["uid"] += "_" + df_expanded.groupby("uid").cumcount().add(1).astype(str)
    df_expanded = df_expanded.fillna(0)
    print(df_expanded)
    X = df_expanded.loc[:, 0:]
    out = Path("output") / str(w)
    out.mkdir(parents=True, exist_ok=True)
    plot_heatmap_plotly(X, np.arange(X.shape[1]), df_expanded["uid"], out, title=f"Samples ({X.shape[0]})", filename=f"samples_{w}.html")

    print("Processing UMAP...")
    df = df_expanded.copy()
    X = np.log(anscombe(df.iloc[:, 2:].values))
    umap_model = umap.UMAP(n_neighbors=200, min_dist=0.1, n_components=2, random_state=42)
    embedding = umap_model.fit_transform(X)

    df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    df_umap["uid"] = df["uid"]
    df_umap["start_date"] = df["start_date"]

    fig = px.scatter(
        df_umap, x="UMAP1", y="UMAP2", color=df_umap["uid"].str.split("_").str[0],
        hover_data=["uid", "start_date"],
        title=f"UMAP Projection of Samples w={w}"
    )

    fig.write_html(out / f"UMAP_{w}.html")

    df = df_expanded.copy()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_pca = X_pca[:,0:2]

    df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
    df_pca["uid"] = df["uid"]
    df_pca["start_date"] = df["start_date"]

    fig = px.scatter(
        df_pca, x="PCA1", y="PCA2", color=df_pca["uid"].str.split("_").str[0],
        hover_data=["uid", "start_date"],
        title=f"2D PCA Projection of Samples w={w}"
    )

    fig.write_html(out / f"2DPCAProjection_{w}.html")

    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    df_knn = pd.DataFrame(X_tsne, columns=["Dim1", "Dim2"])
    df_knn["uid"] = df["uid"]
    df_knn["start_date"] = df["start_date"]

    inertia = []
    k_range = range(1, 100)

    for k in k_range:
        print(f"Elbow Method: {k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_tsne)
        inertia.append(kmeans.inertia_)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=inertia,
        mode='lines+markers',
        marker=dict(color='blue'),
        line=dict(color='blue', width=2),
    ))

    fig.update_layout(
        title=f'Elbow Method for Optimal Clusters w={w}',
        xaxis_title='Number of Clusters',
        yaxis_title='Inertia',
        showlegend=False
    )
    fig.write_html(out / f"elbow_method_plotly_{w}.html")

    optimal_k = 40
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df_knn["cluster"] = kmeans.fit_predict(X_tsne)

    fig = px.scatter(
        df_knn, x="Dim1", y="Dim2", color=df_knn["cluster"],
        hover_data=["uid", "start_date"],
        title=f"KNN-Based t-SNE Projection of Samples w={w}"
    )

    fig.write_html(out / f"KNN-Based-TSNE_{w}.html")

    fig = go.Figure()

    for cluster in range(df_knn["cluster"].nunique()):
        cluster_data = df.iloc[df_knn[df_knn["cluster"] == cluster].index]
        mean_values = cluster_data.iloc[:, 2:].mean(axis=0)

        fig.add_trace(go.Scatter(
            x=np.arange(len(mean_values)),
            y=mean_values,
            mode='lines',
            name=f"Cluster {cluster}",  # Label for the line
        ))

        random_samples = cluster_data.sample(n=2, random_state=42)

        # Create a separate figure for each random sample
        for _, sample in random_samples.iterrows():
            sample_values = sample.iloc[2:].values  # Ignore metadata columns

            # Create a bar plot for the random sample
            fig_ = go.Figure(go.Bar(
                x=np.arange(len(sample_values)),
                y=sample_values,
                name=f"Cluster {cluster} - Sample {sample['uid']}",  # Use sample UID for labeling
            ))
            fig_.update_layout(
                title=f"Sample {sample['uid']} - Cluster {cluster} w={w}",
                xaxis_title="Sample Index",
                yaxis_title="Sample Value"
            )
            fig_.write_html(out / f"sample_{sample['uid']}_cluster_{cluster}_bar_plot_{w}.html")


    # Update layout with title and axis labels
    fig.update_layout(
        title=f"Mean Sample Value Line Plot for Each Cluster w={w}",
        xaxis_title="Sample Index",
        yaxis_title="Mean Value",
        showlegend=True
    )

    fig.write_html(f"mean_line_plot_clusters_plotly_{w}.html")
    fig.show()


if __name__ == "__main__":
    for i in [10]:
        main(i)