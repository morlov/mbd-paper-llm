
## Instalation

conda create -n "mbd_paper"
source activate mbd_paper

pip install -U lightautoml omegaconf pyarrow
mkdir data/embeddings/llm
mkdir data/embeddings/baselines
mkdir logs