# Text Classification using Distributed Semantic Models
This repository is made in lieu with the project for the course Computational Linguistics - 2. 

## How to use
There are two options to run the given codes

### Automated
First, given the script executable permissions
```
chmod +x script.sh
```
Then run the script using
```
./script.sh
```

### Manual
In this case, the required packages need to be installed manually using the command
```
pip install -r requirements.txt
```
Then to run the desired model, use the commands
1. To run the model
```
python3 Code/{Model_Type}/model.py
```
2. To analyse the model's output
```
python3 Code/{Model_Type}/analysis.py
```
The `Model_Type` can be either `Dist_Semantic_Model` or `Naive_Bayes`
> Ensure that the required directories specified in the script are manually made in the correct paths. 