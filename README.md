

Install dependencies (virtualenv is recommended):
```bash
pip install -r requirements.txt 
```

Preprocess the data:
```bash
python preprocess.py --dataset <dataset>
```
where \<dataset> is one of MSL, SMAP or SMD.

To train:
```bash
 python train.py --dataset <dataset>
```









