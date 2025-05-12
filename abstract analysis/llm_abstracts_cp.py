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

def get_features(similarity_within_generation):

  n, n_sen, k = similarity_within_generation.shape[0], similarity_within_generation.shape[1], similarity_within_generation.shape[2]

  degrees_all = np.zeros((n, n_sen, k, 1))
  eigens_all = np.zeros((n, n_sen, k, 5))
  confidence_all = np.zeros((n, n_sen, k, 1))

  for j in range(n_sen):


    D = [np.diag(np.sum(s, axis=1) ) for s in similarity_within_generation[:, j, :, :]]

    D_sqrt_inv = [np.diag(1 / np.sqrt(np.sum(s, axis=1))) for s in similarity_within_generation[:, j, :, :]]
    A = similarity_within_generation[:, j, :, :]
    L = [ np.eye(k) - D_sqrt_inv[i] @ A[i] @ D_sqrt_inv[i] for i in range(n)]

    degrees = np.array([np.sum(D[i]) for i in range(n)])
    degrees = np.repeat(degrees[:, np.newaxis], k, axis=1)
    degrees = degrees[..., np.newaxis]

    eigens = np.array([np.linalg.eigh(L[i])[0][1:6] for i in range(n)])
    eigens = np.repeat(eigens[:, np.newaxis, :], k, axis=1)

    confidence = np.array([np.diag(D[i]) for i in range(n)])
    confidence = confidence[..., np.newaxis]

    degrees_all[:, j, :, :], eigens_all[:, j, :, :], confidence_all[:, j, :, :] = degrees, eigens, confidence

  return degrees_all, eigens_all, confidence_all

def combine_features(inputs, similarity_with_x, degrees, eigens, confidence, pLSI, n_sen, k):


    n = len(inputs)
    X_new = vectorizer.transform(inputs)

    topic_embeddings = transform(pLSI, X_new)



    topic_embeddings_expanded = np.broadcast_to(topic_embeddings[:, None, None, :], (n, n_sen, k, 5))

    combined_features = np.concatenate([topic_embeddings_expanded, similarity_with_x, degrees, eigens, confidence], axis=-1)

    return combined_features

import random

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

import pickle
with open("abstract_data/abstract_input_10000.pickle", "rb") as file:
   input_texts =  pickle.load( file)

with open("abstract_data/abstract_response_10000.pickle", "rb") as file:
   response_texts = pickle.load(file)

#with open("abstract_data/all_response_abstract_9000_new_low_temp.pickle", "rb") as file:
with open("abstract_data/all_response_abstract_9000_new.pickle", "rb") as file:
    all_response = pickle.load(file)

#with open("abstract_data/similarities_abstract_9000_new_low_temp.pickle", "rb") as file:
with open("abstract_data/similarities_abstract_9000_new.pickle", "rb") as file:
    similarities = pickle.load(file)

with open("abstract_data/data_abstract_10000.pickle", "rb") as file:
    articles_selected, topics_selected, articles_additional, topics_additional = pickle.load( file)

generations = []

for i in range(9000):
  cur = []
  for j in range(3):
    cur1 = []
    for k in range(20):
      cur1.append(all_response[i][k][j])
    cur.append(cur1)
  generations.append(cur)
input_texts_join = [". ".join(input_texts[i]) for i in range(9000)]
response_texts_join = [". ".join(response_texts[i]) for i in range(9000)]

scores_y =  np.array([  [similarities[i, j, 2, 3:23] for j in range(3) ] for i in range(9000)])
scores_x =  np.array([  [similarities[i, j, 0:2, 3:23] for j in range(3) ] for i in range(9000)])
scores_within =   np.array([  [similarities[i, j, 3:23, 3:23] for j in range(3) ] for i in range(9000)])
scores_x = scores_x.transpose(0, 1, 3, 2)

import numpy as np
orig_seed = np.random.randint(low = 0, high = 100000, size = 1)
np.random.seed(orig_seed)
indices = np.random.permutation(9000)
aug, eval = indices[:9000], indices[8000:]

training, calibration, test = aug[np.arange(0, 7000)], aug[np.arange(7000, 8500)], aug[np.arange(8500, 9000)]


input_train,  input_calib, input_test = [input_texts[i] for i in training], [input_texts[i] for i in calibration], [input_texts[i] for i in test]
response_train, response_calib, response_test = [response_texts[i] for i in training], [response_texts[i] for i in calibration], [response_texts[i] for i in test]
input_join_train,  input_join_calib, input_join_test = [input_texts_join[i] for i in training], [input_texts_join[i] for i in calibration], [input_texts_join[i] for i in test]
response_join_train, response_join_calib, response_join_test = [response_texts_join[i] for i in training], [response_texts_join[i] for i in calibration], [response_texts_join[i] for i in test]

scores_y_train, scores_y_calibration, scores_y_test = scores_y[training, :, :], scores_y[calibration, :, :],  scores_y[test, :, :]
scores_x_train, scores_x_calibration, scores_x_test = scores_x[training, :, :, :], scores_x[calibration, :, :, :],  scores_x[test, :, :, :]
scores_within_train, scores_within_calibration, scores_within_test = scores_within[training, :, :, :], scores_within[calibration, :, :, :],  scores_within[test, :, :, :]
generations_train, generations_calibration, generations_test = [ generations[i]for i in training],  [ generations[i] for i in calibration], [ generations[i] for i in test]

articles_eval, topics_eval = [articles_selected[i] for i in eval], [topics_selected[i] for i in eval]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(input_join_train)
X = vectorizer.transform(input_join_train + input_join_calib)
X = X.astype(np.float64)

pLSI_model = pLSI(precondition=True, solver="projector")
pLSI_model.fit(X.toarray(), K = 5)

degrees_train, eigens_train, confidence_train = get_features(scores_within_train)
features_train = combine_features(input_join_train, scores_x_train, degrees_train, eigens_train, confidence_train, pLSI_model, 3, 20)

degrees_calib, eigens_calib, confidence_calib = get_features(scores_within_calibration)
features_calib = combine_features(input_join_calib, scores_x_calibration, degrees_calib, eigens_calib, confidence_calib, pLSI_model, 3, 20)


degrees_test, eigens_test, confidence_test= get_features(scores_within_test)
features_test = combine_features(input_join_test, scores_x_test, degrees_test, eigens_test, confidence_test, pLSI_model, 3,20)


X_additional = vectorizer.transform(articles_additional)

additional_embeddings = transform(pLSI_model, X_additional)


X_train = features_train.reshape(-1, 14)
Y_train = scores_y_train.flatten()

X_calib = features_calib.reshape(-1, 14)
Y_calib = scores_y_calibration.flatten()

X_test = features_test.reshape(-1, 14)
Y_test = scores_y_test.flatten()

print("point 1 ok")
import xgboost as xgb
from sklearn.metrics import mean_squared_error

dtrain = xgb.DMatrix(X_train, label=Y_train)
dval = xgb.DMatrix(X_calib, Y_calib)

params = {
    "objective": "reg:squarederror",  # Use "binary:logistic" for classification
    "eval_metric": "rmse",  # Root Mean Squared Error
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 100
}

# Train the model and store evaluation results
evals_result = {}
model_xgb = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    evals=[(dtrain, "Train"), (dval, "Validation")],
    early_stopping_rounds=10,
    verbose_eval=False,
    evals_result=evals_result  # Store evaluation results
)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.linear_model import RidgeCV
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor





alphas = np.logspace(-4, 4, 20)


models = {
    "Linear Regression": LinearRegression()
}



# Fit models
for name, model in models.items():
    model.fit(X_train, Y_train)

# Plotting

train_mse, calib_mse = [], []
for i, (name, model) in enumerate(models.items()):
    y_pred_train = model.predict(X_train)
    y_pred_cali = model.predict(X_calib)
    train_mse.append( mean_squared_error(Y_train, y_pred_train))
    calib_mse.append(mean_squared_error(Y_calib, y_pred_cali))

models["XGBoost"] = model_xgb
print("point 2 ok")

thre = 0.7
cs_train = {}
n = 7000
for model_idx, (name, model) in enumerate(models.items()):
    if name == "XGBoost":
        y_pred_train = model.predict(xgb.DMatrix(X_train))  # Convert to DMatrix
    else:
        y_pred_train = model.predict(X_train)

    y_pred_train_reshape = y_pred_train.reshape(n, 3, 20)
    c = -np.ones(n)

    for idx in range(n):  # avoid overwriting model loop index
        pred_vals = y_pred_train_reshape[idx].flatten()
        true_vals = scores_y_train[idx].flatten()  # assumes same shape

        sorted_pred = np.sort(pred_vals)

        for j, threshold in enumerate(sorted_pred):
            mask = pred_vals >= threshold
            if np.sum(true_vals[mask] < thre) <= 3:
                if j == 0:
                    c[idx] == 0
                else:
                    c[idx] = threshold
                break
        if c[idx] == -1:
            c[idx] = np.max(sorted_pred)  + 1e-8
    cs_train[name] = c

thre = 0.7
cs_calib = {}
n = 1500
for model_idx, (name, model) in enumerate(models.items()):
    if name == "XGBoost":
        y_pred_cali = model.predict(xgb.DMatrix(X_calib))  # Convert to DMatrix
    else:
        y_pred_cali = model.predict(X_calib)

    y_pred_cali_reshape = y_pred_cali.reshape(n, 3, 20)
    c = -np.ones(n)

    for idx in range(n):  # avoid overwriting model loop index
        pred_vals = y_pred_cali_reshape[idx].flatten()
        true_vals = scores_y_calibration[idx].flatten()  # assumes same shape

        sorted_pred = np.sort(pred_vals)

        for j, threshold in enumerate(sorted_pred):
            mask = pred_vals >= threshold
            if np.sum(true_vals[mask] < thre) <= 3:
                if j == 0:
                    c[idx] == 0
                else:
                    c[idx] = threshold
                break
        if c[idx] == -1:
            c[idx] = np.max(sorted_pred)  + 1e-8
    cs_calib[name] = c

quantiles = 0.9
res_cp = {}
fitted_topic = np.argmax(features_test[:, 0, 0, :5], 1)

for i, (name, model) in enumerate(models.items()):
    res_cur = np.zeros((500, 3, 20), dtype=bool)

    if name == "XGBoost":
        y_pred_test = model.predict(xgb.DMatrix(X_test))
    else:
        y_pred_test = model.predict(X_test)

    y_pred_test_reshape = y_pred_test.reshape(500, 3, 20)
    c = cs_calib[name]  # length-500 array of thresholds

    tau = np.quantile(np.concatenate((c, np.array([np.inf]))), quantiles)

    for j in range(500):
        res_cur[j] = y_pred_test_reshape[j] > tau

    res_cp[name] = res_cur

detections = {}
error = {}
truth = (scores_y_test > thre).astype(bool)

for i, (name, model) in enumerate(models.items()):
    detection = res_cp[name].astype(bool)
    false = 0

    for j in range(500):
        detected = detection[j, :, :]
        actual = truth[j, :, :]
        denom = np.sum(detected)
        if denom > 0:
            false_rate = np.sum(detected & ~actual) / denom
            if false_rate > 0.2:
                false += 1
    detections[name] = detection
    detection = np.sum(detection & truth) / np.sum(truth)
    error[name] = false / 500
    print("Type 1 error of " + name + " is " + str(false / 500))
    print("Proportion of detected sentence of " + name + " is " + str(detection))
print("point 3 ok")
p = 5 # X1 dimension (number of words)
p0 = 9 # X0 dimension (number of non-topic related covariates)
K = 5 # W dimenstion (number of topics)
covariate_W = True # do we know true topic proportion
extra_covariate = True # whether we have covariates not related to topics

alpha = 0.1 # 1-coverage
gamma = None # scale parameter in Gaussian kernel, if not specified, QuiKMix automatically optimizes it

features_cp_train = features_train.mean(axis=(1, 2))  # average over axes 1 and 2 (3 and 20)
features_cp_calib = features_calib.mean(axis=(1, 2)) # average over axes 1 and 2 (3 and 20)
features_cp_test = features_test.mean(axis=(1, 2))# average over axes 1 and 2 (3 and 20)

res = {}

for i, (name, model) in enumerate(models.items()):
  y_train = cs_train[name]

  y_calib= cs_calib[name]
  quiKMix = QuiKMixCP(alpha = 0.1, K = K, p = p, rand_seed = 42, estimate_mixture = False)

  Smin = np.min(y_calib)
  Smax = np.max(y_calib)
  quiKMix.train( features_cp_train, features_cp_calib, features_cp_test, y_train, y_calib) # train predictor, estimate topics, optimize gamma
  res[name] = quiKMix.predict(Smin, Smax)


cutoff = res
res_quikmix = {}

for i, (name, model) in enumerate(models.items()):
    res_cur = np.zeros((500, 3, 20), dtype=bool)

    if name == "XGBoost":
        y_pred_test = model.predict(xgb.DMatrix(X_test))
    else:
        y_pred_test = model.predict(X_test)

    y_pred_test_reshape = y_pred_test.reshape(500, 3, 20)

    for j in range(500):
        res_cur[j] = y_pred_test_reshape[j] > cutoff[name][j]

    res_quikmix[name] = res_cur


print("point4 ok")
complete_join_test = [ input_join_test[i] + response_join_test[i] for i in range(500)]

filter0 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if scores_y_test[i, j, k] > 0 ]) for j in range(3) ] for i in range(500)]
filter1 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_cp["Linear Regression"][i, j, k]  ] ) for j in range(3) ] for i in range(500)]
#filter2 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_cp["Polynomial Ridge Regression (deg=2)"][i, j, k]   ]) for j in range(3) ] for i in range(500)]
#filter3 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_cp["MLP Regressor"][i, j, k]  ]) for j in range(3) ] for i in range(500)]
filter4 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_cp["XGBoost"][i, j, k]  ]) for j in range(3) ] for i in range(500)]
filter5 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_quikmix["Linear Regression"][i, j, k]  ] ) for j in range(3) ] for i in range(500)]
#filter6 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_quikmix["Polynomial Ridge Regression (deg=2)"][i, j, k]   ]) for j in range(3) ] for i in range(500)]
#filter7 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_quikmix["MLP Regressor"][i, j, k]  ]) for j in range(3) ] for i in range(500)]
filter8 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if res_quikmix["XGBoost"][i, j, k]  ]) for j in range(3) ] for i in range(500)]
filter9 = [ [ ". ".join([generations_test[i][j][k] for k in range(20) if scores_y_test[i, j, k] > .7  ]) for j in range(3) ] for i in range(500)]


LLM0 = [ ". ".join([complete_join_test[i]] + filter0[i]) for i in range(500)]
LLM1 = [ ". ".join([complete_join_test[i]] + filter1[i]) for i in range(500)]
#LLM2 = [ ". ".join([input_join_test[i]] + filter2[i]) for i in range(500)]
#LLM3 = [ ". ".join([input_join_test[i]] + filter3[i]) for i in range(500)]
LLM4 = [ ". ".join([complete_join_test[i]] + filter4[i]) for i in range(500)]
LLM5 = [ ". ".join([complete_join_test[i]] + filter5[i]) for i in range(500)]
#LLM6 = [ ". ".join([input_join_test[i]] + filter6[i]) for i in range(500)]
#LLM7 = [ ". ".join([input_join_test[i]] + filter7[i]) for i in range(500)]
LLM8 = [ ". ".join([complete_join_test[i]] + filter8[i]) for i in range(500)]
LLM9 = [ ". ".join([complete_join_test[i]] + filter9[i]) for i in range(500)]



#texts = [ input_join_test, complete_join_test, LLM0, LLM1, LLM2, LLM3, LLM4, LLM5, LLM6, LLM7, LLM8, LLM9]
texts = [ input_join_test, complete_join_test, LLM0, LLM1, LLM4,LLM5, LLM8,  LLM9]

#methods = [ "masked input", "complete input", "All", "Linear-S", "Polynomial-S", "MLP-S", "XGB-S", "Linear-M", "Polynomial-M", "MLP-M", "XGB-M", "Oracle"]
methods = [ "masked input", "complete input", "All", "Linear-S",  "XGB-S",  "Linear-M",  "XGB-M", "Oracle"]

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


np.random.seed(1234)

scores = np.zeros((50, len(texts)))


word_scores = np.zeros((50, len(texts)))


for k in range(10):
    for i in range(len(texts)):
        docs = texts[i]

        # Split into two random halves
        f, s = split_documents_in_half_random(docs)

        # Build a shared vocabulary across f + s
        vectorizer = CountVectorizer(stop_words='english')
        vectorizer.fit(f + s)

        X_f = vectorizer.transform(f)
        X_s = vectorizer.transform(s)
        X_f = X_f.astype(np.float64)
        X_s = X_s.astype(np.float64)

        with disable_print():
            model1 = pLSI(precondition=True, solver="projector")
            model1.fit(X_f.toarray(), K = 5)
            # Get the topic-word distribution

            model2 = pLSI(precondition=True, solver="projector")
            model2.fit(X_s.toarray(), K = 5)

            topic_pred_f = model1.W_hat
            topic_pred_s = model2.W_hat


        # Compute distance up to permutation
        scores[k, i] = frobenius_distance_up_to_permutation(topic_pred_f, topic_pred_s)


    print(f"Finished iteration {k}")

np.random.seed(12345)

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


all_res = []

for _ in range(1):

    result = []

    for i in range(len(texts)):

        docs = texts[i]
        vectorizer = CountVectorizer(stop_words='english')
        vectorizer.fit(docs)

        X = vectorizer.transform(docs)
        X = X.astype(np.float64)
        with disable_print():

          model = pLSI(precondition=True, solver="projector")
          model.fit(X.toarray(), K = 5)

        pred = np.argmax(model.W_hat, axis = 1)


        result.append(compute_accuracy_with_permutations([topics_selected[i] for i in test], pred)[0])
    print(_)
    all_res.append(result)


print("point5 ok")


from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

models = []
vectorizers = []


for i in range(len(texts)):

    docs = texts[i]
    vectorizer = CountVectorizer(stop_words='english')
    vectorizer.fit(docs)

    vectorizers.append(vectorizer)

    X = vectorizer.transform(docs)
    X = X.astype(np.float64)
    with disable_print():

      model = pLSI(precondition=True, solver="projector")
      model.fit(X.toarray(), K = 5)
    models.append(model)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np
from imblearn.over_sampling import SMOTE  # <-- Note: from imbalanced-learn package
np.random.seed(12345)

articles_eval = articles_additional
topics_eval = topics_additional

# Parameters
n_splits = 8  # You can change this for more/less splits
errors = [[] for _ in range(len(models))]  # list of errors for each model

# Encode topic labels
le = LabelEncoder()
y_encoded = le.fit_transform(topics_eval)

# Cross-validation
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold_idx, (train_index, test_index) in enumerate(skf.split(articles_eval, y_encoded)):
    print(f"Fold {fold_idx + 1}")
    
    articles_train = [articles_eval[i] for i in train_index]
    articles_test = [articles_eval[i] for i in test_index]
    y_train_encoded = y_encoded[train_index]
    y_test_encoded = y_encoded[test_index]
    
    for i, model in enumerate(models):
        # Get vectorizer and documents
        vectorizer = vectorizers[i]
        docs_train = vectorizer.transform(articles_train).toarray()
        docs_test = vectorizer.transform(articles_test).toarray()

        # Get topic distributions as features from pLSI
        X_train = transform(model, docs_train)
        X_test = transform(model, docs_test)

        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train_encoded)

        # Train XGBoost
        clf = xgb.XGBClassifier(eval_metric='mlogloss')
        clf.fit(X_train_resampled, y_train_resampled)

        # Predict and evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test_encoded, y_pred)
        errors[i].append( acc)  # Error = 1 - accuracy


workspace = {'truth': truth, "scores_y_test": scores_y_test,  "accuracy": errors, 'res_cp': res_cp, 'res_quikmix': res_quikmix, 'scores': scores, 'all_res': all_res, 'fitted_topic': fitted_topic, 'true_topics': [topics_selected[i] for i in test], "additional_embeddings": additional_embeddings, "topics_additional": topics_additional}

with open(f'abstract_data/workspace_{orig_seed}.pkl', 'wb') as f:
    pickle.dump(workspace, f)
