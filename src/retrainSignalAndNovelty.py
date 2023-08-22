import os
import datetime
from BinanceDataDownload import fetch_binance_data
from config import *
from extractLastFeaturesForClfTraining import *
from datalabelling import add_labels
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import joblib
import re


def load_recent_models(clf_dir):
    """Load the most recent SVC and LOF models from the specified directory."""

    # Determine the most recent SVC and LOF models based on their filename prefixes
    recent_svc_file = most_recent_file(clf_dir, "signal_clf_")
    recent_lof_file = most_recent_file(clf_dir, "novelty_detection_")

    if not recent_svc_file or not recent_lof_file:
        print("Could not find models in the specified directory!")
        return None, None

    # Load the models using joblib
    svc_model = joblib.load(recent_svc_file)
    lof_model = joblib.load(recent_lof_file)

    return svc_model, lof_model


def most_recent_file(directory, prefix):
    """Returns the most recently created file in the given directory with the specified prefix."""
    files = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not files:
        return None
    return max([os.path.join(directory, f) for f in files], key=os.path.getctime)


def train_and_save_models(clf_dir, symbol='ETHUSDT', timeframe='1h', retrain_time=datetime.timedelta(weeks=1)):
    recent_svc = most_recent_file(clf_dir, "signal_clf_")
    recent_lof = most_recent_file(clf_dir, "novelty_detection_")

    retrain = False

    for model in [recent_svc, recent_lof]:
        if not model:
            retrain = True
            break

        date_str = re.search(r'(\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2})', model)
        if date_str:
            recent_date = datetime.datetime.strptime(date_str.group(1), '%Y%m%d_%H%M%S')

            # Überprüfen, ob retrain_time seit dem letzten Training vergangen ist
            if datetime.datetime.now() - recent_date > retrain_time:
                retrain = True
                break

    if not retrain:
        print("No retraining required!")
        return

    # Daten von Binance herunterladen
    timerange = TRAIN_WINDOW_SIZE + 2 * max(feature_lookbacks)
    df = fetch_binance_data(symbol=symbol, timeframe=timeframe, timerange=timerange)

    # Features und Labels extrahieren
    df_with_feats = integrate_features_to_df(df)
    df_with_feats_and_labels = add_labels(df_with_feats)

    # Daten vorbereiten
    df_with_feats_and_labels = df_with_feats_and_labels.dropna(subset=['label'])
    X = df_with_feats_and_labels[[col for col in df_with_feats_and_labels.columns if 'Feature' in col]]
    y = df_with_feats_and_labels['label']
    X = X.fillna(0)

    # SVC-Classifier trainieren
    svc_model = train_model_wf(X, y)

    # LOF-Modell trainieren
    lof_model = train_lof_model(X)

    # Beide Modelle speichern
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(svc_model, os.path.join(clf_dir, f"signal_clf_{timestamp}.pkl"))
    joblib.dump(lof_model, os.path.join(clf_dir, f"novelty_detection_{timestamp}.pkl"))
    print(f"Models saved with timestamp {timestamp}")


def train_model_wf(X_train, y_train):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(max_iter=1000000))
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lof_model(X_train, contamination=0.0025):
    lof = LocalOutlierFactor(novelty=True, contamination=contamination)
    lof.fit(X_train)
    return lof


# Beispielaufruf:
test_dir = "testdir"
os.makedirs(test_dir, exist_ok=True)
retrain_time = datetime.timedelta(minutes=1)
#train_and_save_models(clf_dir=test_dir, symbol='ETHUSDT', timeframe='1h', retrain_time=retrain_time)
