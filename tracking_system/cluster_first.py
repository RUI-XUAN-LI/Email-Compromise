import json
import difflib
import os
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def find_all_files(base_dir):
    for root, _, files in os.walk(base_dir):
        for f in files:
            yield os.path.join(root, f)


def difflib_similarity(str1, str2):
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def read_as_info(as_file):
    ip_as_dict = {}
    with open(as_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            ip = record["ip"]
            as_info = record["asn"].split(';')[1] if ';' in record["asn"] else "unknown"
            ip_as_dict[ip] = as_info.strip()
    return ip_as_dict


def read_ip_info(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return [entry['ip'] for entry in json_data]


def assign_as_ids(as_list):
    return {as_name: i for i, as_name in enumerate(as_list)}


def create_similarity_matrix(as_list, as_id_map):

    num_as = len(as_list)
    similarity_matrix = {}

    for i, as1 in enumerate(as_list):
        for j in range(i + 1, num_as):
            as2 = as_list[j]
            similarity = difflib_similarity(as1, as2)
            if similarity >= 0.75:
                similarity_matrix[(i, j)] = 0.3
    return similarity_matrix


def read_json_cluster(name, matched_ip_as, as_id_map, similarity_matrix):

    data = []
    labels = []

    with open(name, 'r') as f:
        records = json.load(f)
        for record in records:
            ip = record["ip"]
            if ip in matched_ip_as:
                ip_data = list(map(float, [
                    record["failsum"], record["failemailsum"],
                    record["faildomainsum"], record["passwordsum"],
                    record["passwordmean"], record["logtimestd"], record["intervalmean"],
                    record["intervalstd"], record["ipcntmean"], record["ipcntstd"],
                    record["userexist"], record["usernonexist"]
                ]))

                time_obj = datetime.strptime(record["logtimemean"], '%Y-%m-%d %H:%M:%S')
                seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
                ip_data.append(seconds)

                as_id = as_id_map[matched_ip_as[ip]]
                as_similarities = [similarity_matrix.get((min(as_id, j), max(as_id, j)), 0)
                                   for j in range(len(as_id_map))]
                ip_data.extend(as_similarities)

                data.append(ip_data)
                labels.append(ip)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled, labels


def run_dbscan(name, feature_data):
    da = os.path.splitext(os.path.basename(name))[0]
    data_scaled, labels = feature_data

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan.fit(data_scaled)

    cluster_labels = dbscan.labels_
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters) - (1 if -1 in cluster_labels else 0)

    silhouette_avg = "N/A"
    if n_clusters > 1:
        silhouette_avg = silhouette_score(data_scaled, cluster_labels)
        print(f"Silhouette Coefficient: {silhouette_avg}")
    else:
        print("Clustering not meaningful (only one cluster).")

    cluster_dict = {}
    for label, cluster_id in zip(labels, cluster_labels):
        cluster_dict.setdefault(cluster_id, []).append(label)

    output_file = f"{da}_clusters.txt"
    with open(output_file, "w") as f:
        f.write(f"Clusters: {n_clusters}, Silhouette: {silhouette_avg}\n")
        for cluster_id, ips in cluster_dict.items():
            f.write(f"Cluster {cluster_id}:\n")
            for ip in ips:
                f.write(f"  {ip}\n")

    print(f"Output written to: {output_file}")


def main(json_file, as_info_file):
    ip_list = read_ip_info(json_file)
    ip_as_dict = read_as_info(as_info_file)

    matched_ip_as = {ip: ip_as_dict[ip] for ip in ip_list if ip in ip_as_dict}
    if not matched_ip_as:
        return

    as_list = list(set(matched_ip_as.values()))
    as_id_map = assign_as_ids(as_list)

    similarity_matrix = create_similarity_matrix(as_list, as_id_map)
    feature_data = read_json_cluster(json_file, matched_ip_as, as_id_map, similarity_matrix)
    run_dbscan(json_file, feature_data)


if __name__ == "__main__":

    json_files = [name for name in find_all_files("<DATA_DIR>") if name.endswith(".json")]
    for file in json_files:
        main(file, "<AS_INFO_FILE>")