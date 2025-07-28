Docker + Windows + Jupyter Notebook:

cpu:
docker run --rm --name jax_jupyter -v /home/hex/jax/:/home/jovyan/jax -p 8888:8888 bytewizard/jax:jupyter jupyter lab

gpu:
docker run --rm --gpus all --name jax_jupyter -v /home/hex/jax/:/home/jovyan/jax -p 8888:8888 bytewizard/jax:jupyter jupyter lab

replace /home/hex/jax