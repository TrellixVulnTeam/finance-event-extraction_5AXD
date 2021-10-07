# Compare 2 kinds of finBERT

Google drive share link:https://drive.google.com/drive/folders/15FWgbN7ihAombUQyGI9L3SYPQUZosRdE?usp=sharing

## 1. load the checkpoint model

Locations: finBERT/notebooks/finbert_training_no_load_weight.ipynb

### Command

```python
AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",cache_dir=None, num_labels=3)
```

### Output

![image-20210917164928963](imgs\image-20210917164928963.png)

## 2. load the financial weight 

Locations: finBERT/notebooks/finbert_training.ipynb

Download the weight from https://github.com/yya518/FinBERT

### Command

```python
AutoModelForSequenceClassification.from_pretrained(lm_path, cache_dir=None, num_labels=3)
```

### Output

![image-load_weight](imgs\image-load_weight.png)

