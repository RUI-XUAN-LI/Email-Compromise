from sklearn.cluster import KMeans
import json
import difflib
import os
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.sparse import dok_matrix

def cluster_and_save(data_scaled, labels, output_file):
    cluster_range = range(10, 100)
    best_score = -1
    best_n_clusters = None
    best_labels = None

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        try:
            cluster_labels = kmeans.fit_predict(data_scaled)
            score = silhouette_score(data_scaled, cluster_labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = cluster_labels
        except Exception:
            continue

    if best_labels is None:
        return

    print(f"best cluster: n_clusters={best_n_clusters}, Silhouette Coefficient={best_score}")

    cluster_result = {}
    for meta, cluster_id in zip(labels, best_labels):
        cluster_result.setdefault(cluster_id, []).append(meta)

    with open(output_file, "w") as f:
        f.write(f"KMeans best cluster: n_clusters={best_n_clusters}\n")
        f.write(f"cluster: {len(cluster_result)}, Silhouette Coefficient: {best_score}\n\n")
        for cid, members in cluster_result.items():
            f.write(f"Cluster {cid} | counts={len(members)}\n")
            for m in members:
                filename = m['source']
                ymd = filename.replace(".json", "")
                f.write(f"  {ymd}  ClusterOriginal={m['original_cluster']}\n")
            f.write("\n")

    print(f"output: {output_file}")


def difflib_similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def assign_as_ids(as_list):
    return {as_name: i for i, as_name in enumerate(as_list)}


def create_sparse_similarity_matrix(as_list, as_id_map):
    num_as = len(as_list)
    similarity_matrix = dok_matrix((num_as, num_as), dtype=np.float32)
    for i, as1 in enumerate(as_list):
        for j, as2 in enumerate(as_list):
            if i < j:
                similarity = difflib_similarity(as1, as2)
                if similarity >= 0.75:
                    similarity_matrix[i, j] = 0.0315
                    similarity_matrix[j, i] = 0.0315
    return similarity_matrix


def find_all_files(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".json"):
                yield os.path.join(root, f)


def load_all_records(base_dir):
    all_records = []
    as_set = set()
    for filepath in find_all_files(base_dir):
        with open(filepath, 'r') as f:
            records = json.load(f)
            for record in records:
                if record.get("Cluster", "-1") == "-1":
                    continue
                as_name = record["as"].split(";")[1].strip() if ";" in record["as"] else record["as"]
                as_set.add(as_name)
                record["__source__"] = os.path.basename(filepath)
                all_records.append(record)
    return all_records, list(as_set)


def parse_features(records, as_id_map, similarity_matrix):
    data = []
    labels = []
    for record in records:
        try:
            features = list(map(float, [
                record["failsum"], record["failemailsum"],
                record["faildomainsum"], record["passwordsum"],
                record["passwordmean"], record["logtimestd"], record["intervalmean"],
                record["intervalstd"], record["ipcntmean"], record["ipcntstd"],
                record["userexist"], record["usernonexist"]
            ]))
            time_obj = datetime.strptime(record["logtimemean"], '%H:%M:%S')
            seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            features.append(seconds)
            as_name = record["as"].split(";")[1].strip() if ";" in record["as"] else record["as"]
            as_id = as_id_map.get(as_name, -1)
            if as_id == -1:
                continue
            as_sim = similarity_matrix[as_id].toarray()[0]
            features.extend(as_sim)
            data.append(features)
            labels.append({
                "source": record["__source__"],
                "original_cluster": record["Cluster"]
            })
        except Exception:
            continue
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, labels


def main(base_dir, output_file):
    all_records, as_list = load_all_records(base_dir)
    if not all_records:
        return
    as_id_map = assign_as_ids(as_list)
    similarity_matrix = create_sparse_similarity_matrix(as_list, as_id_map)
    data_scaled, labels = parse_features(all_records, as_id_map, similarity_matrix)
    cluster_and_save(data_scaled, labels, output_file)


if __name__ == "__main__":
    main("<DATA_DIR>", "kmeans_clusters.txt")