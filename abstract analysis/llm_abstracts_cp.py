
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
