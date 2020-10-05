python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
python3 -m pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html