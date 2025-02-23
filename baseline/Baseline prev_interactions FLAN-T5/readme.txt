train.py and test.py can be used to train or test model on different datasets. Changing anything other than the datasets requires you to make changes in these python files
usage example:
	py train.py amazon_beauty (this will train model and save it in path under models folder inside sub-folder
	named as dataset name sent in arguement when running this script which is amazon_beauty in this case.

	py test.py amazon_beauty (this will perform success rate @ 95%, top-k, ndgc tests and print outputs if the	model is trained and saved by train.py otherwise it will give error.

dataset name in the arguement should be identical to folder name inside dataset folder. you can get exact names by running any script without arguement.

Logs of training are by Transformers library trainer (with tqdm)
Logs of tests are given using print()
Change this according to environment or requirements if needed by making changes in scripts.

I ran this code in Kaggle notebook and then converted to these scripts after working with 1 dataset. I had some environment related problems on windows 11 which i fixed but there might be more according to your environment so do tell me if there is any such problem.

For testing, I randomly sampled from whole dataset because almost all datasets had only 1 (category/type) column in all items csv and even that category was same for all items for most of amazon datasets. So i went with this to keep it consistent. Tell me if we need to change this.