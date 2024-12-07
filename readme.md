# Geometry-Aware Sampling for Class-Incremental Learning

## Reproduce the Experiments

1) Clone the repository
2) Install the requirements
   ```bash
   # create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate
   
   # install the requirements
   pip install -r requirements.txt
   ```

3) Run the experiments
   ```bash
    # run the experiments
    python main.py
    ```

## Results

The results are stored in the `tb_data` directory. You can visualize them using tensorboard.

```bash
tensorboard --logdir tb_data --port 6066
```

Open tensorboard in your browser at `localhost:6066`

## Development / Debugging

We use black for code formatting the code. You can install it using pip and run it on the code.

```bash
black .
```

Please run black before committing your changes. And configure your editor to run black on save.

### Short Introduction to Avalanche

[YouTube: Antonio Carta | "Avalanche: an End-to-End Library for Continual Learning"](https://www.youtube.com/watch?v=n6mykeLdeg0)
