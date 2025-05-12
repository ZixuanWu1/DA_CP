

from plsi import pLSI
from quantileKernelMixCp import quikmix2
from quantileKernelMixCp import utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


from quantileKernelMixCp.utils import *
from quantileKernelMixCp.qkm2 import qkm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class QuiKMixCP:
    def __init__(
            self, 
            alpha: float, 
            K: int, 
            p: int, 
            rand_seed:int,
            Phifn=None, 
            gamma=None,
            gamma_grid=np.logspace(-2,2,50), 
            max_steps = 1000,
            eps1=1e-06, 
            eps2=1e-02, 
            estimate_mixture = True,
            randomize = True
        ):
        self.alpha = alpha
        self.K = K
        self.p = p
        if Phifn is None:
            self.Phifn = lambda x: x[:, :K]
        else:
            self.Phifn = Phifn
        self.gamma = gamma
        self.gamma_grid = gamma_grid
        self.max_steps = max_steps
        self.eps1 = eps1
        self.eps2 = eps2
        self.seed = rand_seed
        self.estimate_mixture = estimate_mixture
        self.randomize = randomize

    def optimize_gamma(self, X, y, PhiX):
        """
        Optimizes gamma via Schwarz information criterion
        """
        csic_list = []
        lambdas_list = []
        for gamma in self.gamma_grid:
            #print(f"Gamma is {gamma}")
            res = qkm(X, y.ravel(), 
                    1-self.alpha, 
                    PhiX, 
                    self.max_steps, 
                    gamma,
                    self.eps1,
                    self.eps2)
            csic_list.append(np.min(res['Csic']))
            lambdas_list.append(res['lambda'][np.argmin(res['Csic'])])
        return csic_list, lambdas_list


    def estimate_mixtures(self, freq_train, freq_calib, freq_test):
        """
        Runs probabilistic latent semantic indexing (pLSI) to 
        estimate latent factors (A) and mixture proportions (W)
        """
        freq = np.vstack([freq_train, freq_calib, freq_test])
        W_hat, A_hat = run_plsi(freq, self.K)

        n_train = freq_train.shape[0]
        n_calib = freq_calib.shape[0]

        W_train = W_hat[:n_train, :]
        W_calib = W_hat[n_train:n_train + n_calib, :]
        W_test = W_hat[n_train + n_calib:, :]
        #W_train, A_train = run_plsi(freq_train, self.K)

        #W_calib, A_calib = run_plsi(freq_calib, self.K)
        #P_calib = get_component_mapping(A_calib.T, A_train.T)
        #W_calib_aligned = W_calib @ P_calib.T

        #W_test, A_test = run_plsi(freq_test, self.K)
        #P_test = get_component_mapping(A_test.T, A_train.T)
        #W_test_aligned = W_test @ P_test.T

        return W_train, W_calib, W_test


    def train(self, X_train, X_calib, X_test, y_train, y_calib):
        """
        Trains predictor, estimate latent mixtures, and 
        search over gamma grid for the optimal gamma
        """
        cv_idx, model_idx = train_test_split(
            np.arange(len(X_train)),
            test_size=0.5,
            random_state=self.seed
        )

        # Step 1: Train predictor
        self.scoresCalib =  y_calib
        self.scoresTest =  None
        self.scoresTrain =  y_train[cv_idx]
        
        #self.scoresCalib = np.abs(np.mean(y_train) - y_calib.ravel())
        #self.scoresTest =  np.abs(np.mean(y_train) - y_test.ravel())
        #self.scoresTrain = np.abs(np.mean(y_train) - y_train[cv_idx].ravel())
        

        # Step 2: Estimate latent mixture proportions
        if self.estimate_mixture:
            print("Estimating with pLSI...")
            self.W_train, self.W_calib, self.W_test = self.estimate_mixtures(
                                                        X_train[cv_idx,:self.p],
                                                        X_calib[:,:self.p],
                                                        X_test[:,:self.p])
        else:
            self.W_train, self.W_calib, self.W_test = X_train[cv_idx,:self.p], X_calib[:,:self.p],X_test[:,:self.p]
        # clr transformation since the data is compositional
        self.X_train_clr = np.apply_along_axis(clr, 1, self.W_train)
        self.X_calib_clr = np.apply_along_axis(clr, 1, self.W_calib)
        self.X_test_clr = np.apply_along_axis(clr, 1, self.W_test)

        p0 = X_train.shape[1]
        if self.p < p0: # there are extra covariates
            self.X_train_clr = np.hstack([self.X_train_clr, X_train[cv_idx,self.p:]])
            self.X_calib_clr = np.hstack([self.X_calib_clr, X_calib[:,self.p:]])
            self.X_test_clr = np.hstack([self.X_test_clr, X_test[:,self.p:]])

            self.X_train_clr = row_standardize(self.X_train_clr)
            self.X_calib_clr = row_standardize(self.X_calib_clr)
            self.X_test_clr = row_standardize(self.X_test_clr)

        # Step 3: Optimize for gamma in the gaussian kernel
        if self.gamma is None:
            Phi_train = self.Phifn(self.W_train)
            csic_list, _ = self.optimize_gamma(self.X_train_clr, self.scoresTrain.ravel(), Phi_train)
            self.gamma = self.gamma_grid[np.argmin(csic_list)]
            #print(f"Optimal gamma is {self.gamma}.")

        

    def fit(self):
        """
        Get coverage for each test point
        """
        Phi_calib = self.Phifn(self.W_calib)
        Phi_test = self.Phifn(self.W_test)
        
        covers_rand = []
        covers = []
        for m, (x_val, y_val) in enumerate(zip(self.X_test_clr, self.scoresTest)):
            x_val = x_val.reshape(1, -1)
            X = np.vstack([self.X_calib_clr, x_val])
            y = np.append(self.scoresCalib, y_val)
            PhiX = np.vstack([Phi_calib, Phi_test[m].reshape(1, -1)])
            res = qkm(X, y.ravel(), 
                    1-self.alpha, 
                    PhiX, 
                    self.max_steps, 
                    self.gamma,
                    self.eps1,
                    self.eps2)
            opt = np.argmin(res['Csic'])
            theta_est = res['theta'][opt]
            u = np.random.uniform(-self.alpha,1-self.alpha,size=1)
            covers_rand.append(theta_est[-1] < u)
            covers.append(res['fit'][opt][-1])
            #covers.append(theta_est[-1] < 1-self.alpha)
        return covers_rand, covers


    def predict(self, Smin, Smax, max_iter=100, tol=1e-2):
        Phi_calib = self.Phifn(self.W_calib)
        Phi_test = self.Phifn(self.W_test)

        all_lengths = []
        covers = []
        init_Smin, init_Smax = Smin, Smax

        for m, x_val in tqdm(enumerate(self.X_test_clr), total=len(self.X_test_clr), desc="Predicting"):
            current_Smin, current_Smax = init_Smin, init_Smax
            iter_count = 0

            u = np.random.uniform(-self.alpha,1-self.alpha,size=1)
            threshold = u if self.randomize else 1-self.alpha

            x_val = x_val.reshape(1, -1)
            X = np.vstack([self.X_calib_clr, x_val])
            PhiX = np.vstack([Phi_calib, Phi_test[m].reshape(1, -1)])
        
            while (current_Smax - current_Smin) > tol and iter_count < max_iter:
                Smed = (current_Smax + current_Smin) / 2.0
                y = np.append(self.scoresCalib, Smed)
                res = qkm(X, y.ravel(), 
                        1 - self.alpha, 
                        PhiX, 
                        self.max_steps, 
                        self.gamma,
                        self.eps1,
                        self.eps2)
                opt = np.argmin(res['Csic'])
                theta_est = res['theta'][opt]
                score_diff = theta_est[-1] - threshold
                fit = res['fit'][opt][-1]

                if score_diff > 0:
                    current_Smax = Smed
                else:
                    current_Smin = Smed

                iter_count += 1

            all_lengths.append(fit)
        return all_lengths


def split_documents_in_half_random(documents):
    first_halves = []
    second_halves = []
    for doc in documents:
        words = doc.split()
        random.shuffle(words)  # shuffle words in-place
        mid = len(words) // 2
        first_halves.append(" ".join(words[:mid]))
        second_halves.append(" ".join(words[mid:]))
    return first_halves, second_halves
import numpy as np
from scipy.linalg import svd

import numpy as np
from scipy.optimize import linear_sum_assignment

def frobenius_distance_up_to_permutation(W1, W2):
    # Step 1: Compute the cost matrix: squared Frobenius norm of column differences
    n = W1.shape[1]
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cost_matrix[i, j] = np.linalg.norm(W1[:, i] - W2[:, j])**2

    # Step 2: Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Step 3: Permute W2 columns and compute the Frobenius norm of the difference
    W2_perm = W2[:, col_ind]
    return np.linalg.norm(W1 - W2_perm, ord='fro')


from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score
import sys
import contextlib
import io
def get_top_words(lda_model, vectorizer, top_n=10):
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_words = []
    for topic_weights in lda_model.components_:
        top_indices = topic_weights.argsort()[::-1][:top_n]
        top_words.append(set(feature_names[top_indices]))
    return top_words

def topic_word_jaccard_distance(top_words1, top_words2):
    # Brute-force match topics using Jaccard similarity
    n_topics = len(top_words1)
    cost_matrix = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(n_topics):
            intersection = len(top_words1[i] & top_words2[j])
            union = len(top_words1[i] | top_words2[j])
            cost_matrix[i, j] = 1 - intersection / union if union > 0 else 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].mean()

@contextlib.contextmanager
def disable_print():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout  # This automatically happens when exiting the with block

def transform(model, X_new):
    """
    Project new documents into the topic space using least-squares regression with simplex projection.

    Parameters
    ----------
    X_new : array-like, shape (n_docs, n_words)
        New document-term matrix (same vocab and preprocessing as training data).

    Returns
    -------
    W_new : array-like, shape (n_docs, n_topics)
        Estimated topic distributions for new documents.
    """

    A = model.A_hat  # shape: (K, p)
    AAt_inv = np.linalg.inv(A @ A.T + 1e-8 * np.eye(A.shape[0]))  # shape: (K, K)
    projector = A.T @ AAt_inv         # shape: (p, K)

    # Estimate W
    W_est = X_new @ projector  # shape: (n_docs, K)

    # Project rows to simplex
    W_proj = np.array([model._euclidean_proj_simplex(w) for w in W_est])
    return W_proj
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple
import pandas as pd

def compute_accuracy_with_permutations(
    true_labels: List[str],
    predicted_labels: List[str]
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    """
    Compute overall and per-category accuracy of topic assignments considering permutations of topics.

    Parameters:
    - true_labels: List of strings representing ground truth topic labels for documents
    - predicted_labels: List of strings representing predicted topic assignments for documents

    Returns:
    - overall_accuracy: float
        Overall accuracy after aligning predicted labels to true labels
    - category_accuracies: Dict[str, float]
        Dictionary mapping each true category to its accuracy
    - df_aligned_confusion: pd.DataFrame
        Aligned confusion matrix
    """
    # Step 1: Map labels to indices
    true_categories = set(true_labels)
    predicted_categories = set(predicted_labels)

    label_to_index_true = {label: i for i, label in enumerate(true_categories)}
    label_to_index_predict = {label: i for i, label in enumerate(predicted_categories)}

    # Convert labels to numeric indices
    true_numeric = [label_to_index_true[label] for label in true_labels]
    predicted_numeric = [label_to_index_predict[label] for label in predicted_labels]

    # Step 2: Build confusion matrix
    n_classes = len(true_categories)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_numeric, predicted_numeric):
        confusion_matrix[t, p] += 1

    # Step 3: Find optimal alignment
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Step 4: Compute aligned predictions
    aligned_predictions = np.array([mapping[p] for p in predicted_numeric])

    # Step 5: Calculate overall accuracy
    correct_predictions = [a == t for a, t in zip(aligned_predictions, true_numeric)]
    overall_accuracy = sum(correct_predictions) / len(true_numeric)

    # Step 6: Calculate accuracy for each true category
    category_accuracies = {}
    for label, idx in label_to_index_true.items():
        aligned_idx = mapping.get(idx, idx)
        relevant_docs = [i for i, t in enumerate(true_numeric) if t == idx]
        if relevant_docs:
            category_correct = sum(aligned_predictions[i] == idx for i in relevant_docs)
            category_accuracies[label] = round(category_correct / len(relevant_docs), 3)
        else:
            category_accuracies[label] = 0.0

    # Step 7: Aligned confusion matrix
    cate = [label for label in true_categories]
    confusion_matrix = confusion_matrix[:, col_ind]
    df_aligned_confusion = pd.DataFrame(confusion_matrix, index=cate, columns=cate)

    return overall_accuracy, category_accuracies, df_aligned_confusion
