# should be *sourced* in this folder
cd .. # now we are in project root
conda activate chain_control

if ! command -v nvidia-smi &> /dev/null
then
    pip install --upgrade jax jaxlib
fi
if command -v nvidia-smi &> /dev/null
then
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
fi

pip install -e .
yes | pip uninstall dm_control
yes | pip install dm_control
