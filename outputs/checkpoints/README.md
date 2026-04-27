Trained Model Weights

Each file here is a PyTorch `state_dict` saved with `torch.save(model.state_dict(), ...)`



###Train from scratch

```bash
python scripts/download_data.py

python scripts/train_all.py --epochs 5 --batch-size 8

You can train a subset only:

```bash
python scripts/train_all.py --models spatial frequency
```