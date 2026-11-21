# Importing the dependencies
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree
from scipy.spatial.distance import squareform


class HierarchicalCluster:

    @staticmethod    
    def jaccard_distance(X):
        
        """
        Calculates the pairwise Jaccard distance matrix for rows in X,
        ignoring NaN values pairwise.

        Parameters
        ----------
        X: Feature matrix (NumPy array, list of lists, or Pandas DataFrame)
        """

        X_arr = np.asarray(X, dtype=float)

        # Track non-NaN entries (1 if valid, 0 if NaN)
        valid = (~np.isnan(X_arr)).astype(int)

        # Binarize positive features while treating NaNs as 0
        X_bin = np.where(np.isnan(X_arr), False, X_arr.astype(bool)).astype(int)

        # Intersection: number of shared positive features
        intersection = X_bin @ X_bin.T

        # Count 1s in row i where row j is ALSO valid (and vice versa)
        ones_i_valid_j = X_bin @ valid.T
        ones_j_valid_i = ones_i_valid_j.T

        # Union across mutually valid features
        union = ones_i_valid_j + ones_j_valid_i - intersection

        # Calculate Jaccard similarity
        similarity = np.divide(intersection, union, out=np.zeros_like(union, dtype=float), where=union != 0)

        # Both all-zero vectors are identical
        similarity[union == 0] = 1.0

        # Convert similarity into distance
        distance = 1.0 - similarity

        # Ensure exact zeros on the diagonal (prevents floating-point artifacts)
        np.fill_diagonal(distance, 0.0)

        return distance


    @staticmethod    
    def k_split_cluster(X, method: str, n_clusters):

        """
        This function split the data into "k" clusters based on the inputs. It acts as Hierarchical Clustering.
        
        Paremeters
        ----------

        X: is the feature Matrix

        method: The method correspoding with SciPy.linkage(). It can be: "average", "single","complete" etc.

        n_clusters: The number of clusters the method should be creating
                
        """

        # calculating the Jaccard Distance
        D = HierarchicalCluster.jaccard_distance(X)

        # Since distance matrix is already calculated. It must be converted into condensed distance matrix with squareform()
        condensed_D = squareform(D)

        # Perform hierarchical clustering 
        Z = linkage(condensed_D, method=method)
        labels = cut_tree(Z, n_clusters=n_clusters).flatten()

        return labels


    @staticmethod    
    def silhouette_score(D, labels):

        """
        This function calculates the silhouette score of cluster

        Parameters
        ----------

        D: The Jaccard distance of products
        
        labels: The list of clusters created via the clustering algorithm
        
        """

        unique_labels = np.unique(labels)

        silhouettes = []

        for i in range(len(labels)):
            # find other products in the same cluster
            same_cluster = labels == labels[i]
            # exclude the product itself
            same_cluster[i] = False

            # Singleton clusters have silhouette 0
            if not np.any(same_cluster):
                silhouettes.append(0)
                continue

            # Average distance to own cluster
            a = np.mean(D[i, same_cluster])

            # Calculate average distance to every other cluster
            other_distances = []

            for cluster in unique_labels:
                if cluster == labels[i]:
                    continue

                other_cluster = labels == cluster
                avg_distance = D[i, other_cluster].mean()
                other_distances.append(avg_distance)

            # minimum average distance to other cluster
            b = min(other_distances)
            
            # Calculate silhouette and append it 
            s = (b - a) / max(a, b)
            silhouettes.append(s)
            
        return np.mean(silhouettes)
