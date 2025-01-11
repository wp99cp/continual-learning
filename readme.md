# Geometry-Aware Sampling for Class-Incremental Learning

## Reproduce the Experiments

To reproduce the results you need a GPU with at least 16GB of memory.

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

## Hyperparameters

Inside `main.py` you find multiple parameters you can tweak to select which experiments get executed, what dataset is
used. Essential hyperparameters such as `batch_size`, `number of tasks` etc. can also be tuned directly within
`main.py`.

Additional experiment specific hyperparameters can be found in the corresponding experiment classes. Essential
hyperparameters that are shared across all experiments can be found on `experiments/base_experiment.py`, here
the `lr`, optimizer, ... are defined and can be changed.

## Results

The results are stored in the `tb_data` directory. You can visualize them using tensorboard.

```bash
tensorboard --logdir tb_data --port 6066
```

Open tensorboard in your browser at `localhost:6066`. Now you should be able to see all the results and plots used in
the paper.

### Regenerate Result Figures from Data

All metrics returned are saved inside a `.pkl` file. You can regenerate the figures using
the following command:

```bash
python main.py --res_path tb_data/<date-of-run>_results
```

The results are then saved inside a tensorboard directory. You can visualize them using tensorboard.

```bash
tensorboard --logdir tb_data/<date-of-run>_results --port 6066
```

## Development / Debugging

We use black for code formatting the code. You can install it using pip and run it on the code.

```bash
black .
```

Please run black before committing your changes. And configure your editor to run black on save.

### Short Introduction to Avalanche

[YouTube: Antonio Carta | "Avalanche: an End-to-End Library for Continual Learning"](https://www.youtube.com/watch?v=n6mykeLdeg0)
