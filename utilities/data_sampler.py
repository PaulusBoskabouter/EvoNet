import numpy as np
from sklearn.model_selection import train_test_split
from objects.Functions import SpikeFunc


def get_train_val_test(spike:SpikeFunc, seed:int=42, num_samples:int=10000):
    

    X = np.random.uniform(low=-spike.span * 1.25, high=spike.span * 1.25, size=num_samples)
    Y = np.array([spike.func(x) for x in X])

    X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size=0.3,
    random_state=seed
    )
    print(f"train size:\t{len(X_train)}")
    print(f"val size:\t{len(X_val)}")

    return (X_train, Y_train), (X_val, Y_val)