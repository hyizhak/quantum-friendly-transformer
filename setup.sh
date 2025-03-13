# create conda environment
env_name='q-transformer'
conda env update -n $env_name -f reqs/conda.yaml

# activate conda environment within this script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name

# install some dependencies
pip install -r reqs/requirements.txt