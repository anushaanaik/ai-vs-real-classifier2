# Dataset

The dataset is **not included** in this repository due to file size.

## Download

See `dataset_link.txt` for the Kaggle link.

After downloading, place the zip at:

```
data/ai-generated-images-vs-real-images.zip
```

Then run training:

```bash
python src/train.py --zip data/ai-generated-images-vs-real-images.zip
```

## Expected Structure Inside Zip

```
ai-generated-images-vs-real-images/
├── real/
│   ├── image_001.jpg
│   └── ...
└── fake/        # or 'ai' / 'artificial'
    ├── image_001.jpg
    └── ...
```

The `data_loader.py` auto-detects class folders by name.
