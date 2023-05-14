# NeurIPS_2023
For Reviewers

Hello,

This repo contains all the code from our experiments.

All you need to do is:

```
conda create --name semifactual
conda activate semifactual
conda install -c anaconda pandas
conda install -c anaconda seaborn
conda install -c anaconda scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge jsonschema
conda install -c conda-forge imbalanced-learn
conda install pytorch::pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge tensorboard
conda install -c conda-forge cvxpy
```

For the non-causal tests go into one dataset's folder and run
```
python main.py 'dataset_name'
```
And the results will reproduce. 

For the causal tests just run
```
python run_benchmarks.py
```

And again the results will reproduce.

Thanks,
Anon Author(s)
