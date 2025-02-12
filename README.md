
# Multimodal Banking Dataset Paper LLM Experiments

## Create environment

```console
conda create -n "mbd_paper"
source activate mbd_paper
pip install requirements.txt
```

## Setup some directories

```console
mkdir data/embeddings/llm
mkdir data/embeddings/baselines
mkdir logs
```

## Run downstream for llm
```console
bash script/run-all-llm.sh
```

## Run downstream for baselines
```console
bash script/run-all-baseline.sh
```