# msml605-face-ID
# stuff to pip install(later add to requirements.txt):

pip install --upgrade tensorflow-datasets
# -----------------------------------------------------------------------------

# Before we create a venv make sure you have Python 3.11.14 or python 3.11 installed (any other version will not download the dataset)



## creating a venv:
# in powershell
py -m venv .venv

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

.\.venv\Scripts\Activate.ps1

# in MAC
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# to verify:
pip list
# -----------------------------------------------------------------------------


# To run and load the dataset we have to run the following command:
python scripts/ingest_dataset.py --config configs/m1.yaml

            OR
            
python3 scripts/ingest_dataset.py --config configs/m1.yaml

# run to create pairs 
python3 scripts/create_pairs.py --config configs/m1.yaml

# to run the benchmark file that compares euclidean and cosine:

python3 scripts/benchmark.py --config configs/m1.yaml

# To run the tests 
python3 -m pytest tests/ -v
