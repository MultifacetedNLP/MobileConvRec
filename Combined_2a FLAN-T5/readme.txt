You have to run train.py script first which will train and save the model and running test.py will
test the model and give test results.
usage:
    python train.py dataset_name modelA_path modelB_path
    python test.py dataset_name modelA_path modelB_path

    model A is conversation only flan-t5
    model B is interaction only flan-t5

dataset name in the arguement should be identical to folder name inside dataset folder. you can get exact names by running any script without arguement.

Logs of training are by Transformers library trainer (with tqdm)
Logs of tests are given using print()
Change this according to environment or requirements if needed by making changes in scripts.