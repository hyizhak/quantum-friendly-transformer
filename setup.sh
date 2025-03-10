# create conda environment
env_name='q-transformer'
conda env update -n $env_name -f reqs/conda.yaml
conda activate $env_name

# install some dependencies
pip install torch 
pip install sklearn
pip install genomic-benchmarks
