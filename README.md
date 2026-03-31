# The Geometry of Representational Failures in Vision Language Models

### Probe-based concept vectors

Generate dataset:
```bash
python -m generate_datasets dataset=cs_bal_1t
python -m generate_datasets dataset=cs_bal_1t_test
```

Collect hidden layer activations:
```bash
python -m generate_activations --multirun +experiment=bal_qwen7b dataset=cs_bal_1t,cs_bal_1t_test
```

Train probes:
```bash
python -m train_probes +experiment=bal_qwen7b probe=attn_single dataset=cs_bal_1t
```

Evaluate probes performance on both training and test sets:
```bash
python -m evaluate_probes_batch +experiment=bal_qwen7b probe=attn_single dataset=cs_bal_1t
```

Inspect and normalize vectors: notebook `4_2_a_probe.ipynb`

### Centroid-based concept vectors

Generate activations:
```bash
python -m generate_contrastive_activations.py +experiment=bal_qwen7b
```

Extract concept vectors: notebook `4_2_b_centroid.ipynb`

### Real world dataset steering
Notebook `4_1_realsteer.ipynb`

### Synthetic dataset steering
Generate dataset:
```bash
python -m evaluate_temp +experiment=bal_qwen7b
```

Perform steering;
```bash
python -m evaluate_temp_steer +experiment=bal_qwen7b template.name=[probe|probepca|centroid]
```

### Visual search

Generate dataset (if not already present):
```bash
python -m generate_datasets dataset=cs_bal_1t
```

Collect model outputs AND hidden layer activations:
```bash
python -m generate_activations +experiment=bal_qwen7b
```


