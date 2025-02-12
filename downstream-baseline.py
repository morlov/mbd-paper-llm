import argparse
import logging

from warnings import simplefilter

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task



from omegaconf import OmegaConf

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str)

PCA_DIM = 256 # Dimension of embeddings after pca
TASK_TYPE = 'binary' # Automl Task Type
DROP_COLS = ['client_id', 'fold']
TARGETS = ['bcard_target', 'cred_target', 'zp_target', 'acquiring_target']
N_TARGETS = len(TARGETS)
N_FOLDS  = 5
RANDOM_STATE = 42



def apply_pca(df_train, df_test):
    
    logger.info("Applying PCA...")

    features_cols = list(filter(lambda x: x.startswith('embed_'), df_train.columns))

    pca = PCA(n_components=PCA_DIM)
    X_train = pca.fit_transform(df_train[features_cols])
    X_test = pca.transform(df_test[features_cols])

    df_train.drop(features_cols, axis=1, inplace=True)
    df_test.drop(features_cols, axis=1, inplace=True)
    
    for i in tqdm(range(PCA_DIM)):
        df_train.loc[:, f"embed_{i}"] = X_train[:, i]
        df_test.loc[:, f"embed_{i}"] = X_test[:, i]
        
    logger.info("PCA done.")
        
    return df_train, df_test

def validate_embeddings(df_train, df_test, target, conf):
    
    task = Task(TASK_TYPE)
    drop_targets = [t for t in TARGETS if t != target]
    roles = {
        'target': target,
        'drop': DROP_COLS + drop_targets
    }
        
    automl = TabularAutoML(
        task = task, 
        timeout = conf['timeout'],
        cpu_limit = conf['n_threads'],
        reader_params = {'n_jobs': conf['n_threads'], 'cv': conf['n_folds'], 'random_state': RANDOM_STATE}
    )


    automl.fit_predict(df_train, roles = roles, verbose = 1)
    pred = automl.predict(df_test) 
    return roc_auc_score(df_test[target].values, pred.data[:, 0])

def run_and_report(conf):
    
    aucs_fold = np.zeros([N_FOLDS, N_TARGETS])
    rename_columns = {'target_1': 'bcard_target', 'target_2': 'cred_target', 'target_3': 'zp_target', 'target_4': 'acquiring_target'}

    for fold in tqdm(range(N_FOLDS)):

        emb_path = f'{conf["embedding_path"]}/fold={fold}/dataset.parquet'
        logger.info(f"Read targets from: {conf['target_path']}")
        df_targets = pd.read_parquet(conf['target_path'])
        logger.info(f"Read embeddings from: {emb_path}")
        df_emb = pd.read_parquet(emb_path)
        logger.info("Merge targets and embeddings")
        df = df_targets.merge(df_emb, on="client_id").fillna(0) # For geo dataset
        logger.info(f"Merged. Embeeding size: {df_emb.shape[0]}, target size: {df_targets.shape[0]}, merged size: {df.shape[0]}")
        
        df_test = df[df['fold']==fold]
        df_train = df[df['fold']!=fold]

        if conf.apply_pca:
            df_train, df_test = apply_pca(df_train.copy(), df_test.copy())
        
        for i, t in enumerate(TARGETS):
            logger.info(f"Running fold: {fold+1}/{N_FOLDS}, target: {i+1}/{N_TARGETS}.")
            aucs_fold[fold, i] = validate_embeddings(df_train, df_test, t, conf['automl_settings'])

    means = np.mean(aucs_fold, axis=0)
    stds = np.std(aucs_fold, axis=0)

    global_mean = np.mean(np.mean(aucs_fold, axis=1))
    global_std = np.std(np.mean(aucs_fold, axis=1))

    report_file = conf["results_dir"] + conf['experiment_name'] + '.txt'

    logger.info(f"Experiments completed, writing results to {report_file}")
    
    with open(report_file, "w") as f:
        f.write(f"Experiment: {conf['experiment_name']}\n\n")

        f.write(f"{pd.DataFrame(aucs_fold, columns=TARGETS).to_string()}\n\n")
        
        for t, (m, s) in enumerate(zip(means, stds)):
            result = f"Target: {TARGETS[t]}, ROC AUC: {m:.3f}+-{s:.3f}"
            logger.info(result) 
            f.write(result + "\n")
        f.write(f"\nROC AUC: {global_mean:.3f}+-{global_std:.3f}"   + "\n")

def main(args): 
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    logging.basicConfig(filename=conf['log_dir'] + conf['experiment_name'] + '.log', filemode='w', level=logging.INFO)
    logger.info(f"Starting experiments.")
    run_and_report(conf)
    logger.info(f"Done.")


if __name__ == "__main__":
    main(parser.parse_args())
