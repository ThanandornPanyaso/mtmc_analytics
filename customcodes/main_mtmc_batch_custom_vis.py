# **Copyright (c) 2009-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.**
import os
import argparse
import logging
import numpy as np
from sklearn.cluster import HDBSCAN
from mdx.mtmc.schema import Frame
from mdx.mtmc.core.data import Loader, Preprocessor
from mdx.mtmc.utils.io_utils import validate_file_path, load_json_from_file_line_by_line, load_json_from_file
from mdx.mtmc.config import AppConfig
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%y/%m/%d %H:%M:%S", level=logging.INFO)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP

class MTMCFeatureClusteringApp:
    """
    Controller module for MTMC clustering based on re-id features only.

    :param str config_path: path to the app config file
    """

    def __init__(self, config_path: str) -> None:
        # Make sure the config file exists
        valid_config_path = validate_file_path(config_path)
        if not os.path.exists(valid_config_path):
            logging.error(f"ERROR: The indicated config file `{valid_config_path}` does NOT exist.")
            exit(1)

        self.config = AppConfig(**load_json_from_file(config_path))
        logging.info(f"Read config from {valid_config_path}\n")
        self.loader = Loader(self.config)
        self.preprocessor = Preprocessor(self.config)

    def extract_embeddings(self, frames: list[Frame]) -> np.array:
        """
        Extract embeddings from frames.

        :param list[Frame] frames: list of frames containing object data
        :return: array of embeddings
        :rtype: np.array
        """
        embeddings = []
        for frame in frames:
            for obj in frame.objects:
                if obj.embedding is not None:
                    embeddings.append(np.array(obj.embedding))
        if not embeddings:
            logging.error("ERROR: No embeddings found in the data.")
            exit(1)
        behaviors = self.preprocessor.normalize_embeddings(embeddings)
        return np.vstack(behaviors)

    def cluster_embeddings(self, embeddings: np.array) -> np.array:
        """
        Cluster embeddings using HDBSCAN.

        :param np.array embeddings: array of embeddings
        :return: cluster labels
        :rtype: np.array
        """
        logging.info("Clustering embeddings...")
        clustering_model = HDBSCAN(min_cluster_size=self.config.clustering.hdbscanMinClusterSize)
        labels = clustering_model.fit_predict(embeddings)
        logging.info(f"Clustering complete. Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
        return labels

    def start_clustering(self) -> None:
        # Load input data
        frames = None
        json_data_path = self.config.io.jsonDataPath
        if os.path.isfile(json_data_path):
            frames = self.loader.load_json_data_to_frames(json_data_path)
        else:
            logging.error(f"ERROR: The JSON data path {json_data_path} does NOT exist.")
            exit(1)

        logging.info("Data loading complete\n")

        # Extract embeddings
        embeddings = self.extract_embeddings(frames)

        # Perform clustering
        labels = self.cluster_embeddings(embeddings)

        # Visualize clustering
        self.visualize_clustering(embeddings, labels, method="tsne")

        # Save results
        output_path = self.config.io.outputDirPath
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, "clustering_results.json")

        results = [{"embedding_index": idx, "cluster_id": int(label)} for idx, label in enumerate(labels)]
        with open(output_file, "w") as f:
            for result in results:
                f.write(f"{result}\n")

        logging.info(f"Clustering results saved to {output_file}\n")


    def visualize_clustering(self, embeddings: np.array, labels: np.array, method: str = "tsne") -> None:
        """
        Visualize the clustering result in a 2D plot.

        :param np.array embeddings: array of embeddings
        :param np.array labels: cluster labels
        :param str method: dimensionality reduction method ("tsne" or "umap")
        :return: None
        """
        logging.info(f"Visualizing clustering using {method}...")
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "umap":
            reducer = UMAP(n_components=2, random_state=42)
        else:
            logging.error("ERROR: Unsupported visualization method. Use 'tsne' or 'umap'.")
            return

        reduced_embeddings = reducer.fit_transform(embeddings)

        # Plot the results
        plt.figure(figsize=(10, 8))
        unique_labels = set(labels)
        for label in unique_labels:
            mask = labels == label
            color = "gray" if label == -1 else None  # Use gray for noise
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], label=f"Cluster {label}", s=10, alpha=0.7, color=color)
        plt.legend(loc="best", fontsize="small", markerscale=2)
        plt.title(f"Clustering Visualization ({method.upper()})")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=validate_file_path, default="resources/app_mtmc_config.json",
                        help="The input app config file")
    args = parser.parse_args()
    clustering_app = MTMCFeatureClusteringApp(args.config)
    clustering_app.start_clustering()
