# msml605-face-ID
# stuff to pip install(later add to requirements.txt):

pip install --upgrade tensorflow-datasets
# creating a venv in powershell:
py -m venv .venv

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# to verify:
pip list

# To run and load the dataset we have to run the following command:
python scripts/ingest_dataset.py --config configs/m1.yaml

            OR
            
python3 scripts/ingest_dataset.py --config configs/m1.yaml

# run to create pairs 
python scripts/create_pairs.py --config configs/m1.yaml
