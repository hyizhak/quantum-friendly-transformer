# create conda environment
env_name='q-transformer'
conda env update -n $env_name -f reqs/conda.yaml
conda init bash
source ~/.bashrc
conda activate $env_name

# install some dependencies
pip install torch 
pip install scikit-learn
pip install genomic-benchmarks
pip install transformers datasets tokenizers accelerate