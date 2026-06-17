from math import ceil
from typing import Callable, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

Op = Callable[[Tensor, Tensor, int], tuple[Tensor, Tensor, Tensor]]


def prime_factors(n: int) -> list[int]:
    factors = []
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            factors.append(divisor)
            while n % divisor == 0:
                n //= divisor
        divisor += 1
    if n > 1:
        factors.append(n)
    return factors


def primitive_root(prime: int) -> int:
    if prime < 3:
        raise ValueError("permuted_quadratic requires an odd prime")

    phi = prime - 1
    factors = prime_factors(phi)
    for candidate in range(2, prime):
        if all(pow(candidate, phi // factor, prime) != 1 for factor in factors):
            return candidate

    raise ValueError(f"could not find a primitive root modulo {prime}")


def permuted_quadratic_task(x: Tensor, y: Tensor, p: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    g^q mod p, where g is a primitive root and q is quadratic in x and y.

    The rule is compact in exponent space, but the primitive-root permutation
    makes adjacent labels look irregular, which makes validation generalization
    harder than the smooth arithmetic tasks.
    """
    root = primitive_root(p)
    exponent = (x * x + x * y + 3 * y * y + 5 * x + 7 * y) % (p - 1)
    labels = torch.tensor(
        [pow(root, int(value), p) for value in exponent.tolist()],
        dtype=x.dtype,
    )
    return x, y, labels


DIVISION_MODULO_OPERATIONS: dict[str, Op] = {
    "x/y": lambda x, y, p: (x * y % p, y, x),
}

ALL_MODULO_OPERATIONS: dict[str, Op] = {
    "x+y": lambda x, y, p: (x, y, (x + y) % p),
    "x-y": lambda x, y, p: (x, y, (x - y) % p),
    "permuted_quadratic": permuted_quadratic_task,
    **DIVISION_MODULO_OPERATIONS,
}

ALL_OPERATIONS: dict[str, Op] = {
    **ALL_MODULO_OPERATIONS,
}


def operation_mod_p_data(
    operation: str, p: int, eq_token: int, op_token: int
) -> tuple[Tensor, Tensor]:
    """
    x◦y (mod p) for 0 <= x < p, 1 <= y < p if operation in DIVISION_MODULO_OPERATIONS
    x◦y (mod p) for 0 <= x, y < p otherwise
    """
    x_vals = torch.arange(0, p)
    y_vals = torch.arange(0 if operation not in DIVISION_MODULO_OPERATIONS else 1, p)
    x, y = torch.cartesian_prod(x_vals, y_vals).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    x, y, labels = ALL_OPERATIONS[operation](x, y, p)

    inputs = torch.stack([x, op, y, eq], dim=1)

    return inputs, labels


def get_data_loaders(
    operation: str, prime: int, training_fraction: float, batch_size: int
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    inputs, labels = operation_mod_p_data(operation, prime, prime, prime + 1)
    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    batch_size = min(batch_size, ceil(len(dataset) / 2))

    train_loader = cast(
        DataLoader[tuple[Tensor, Tensor]],
        torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    )
    val_loader = cast(
        DataLoader[tuple[Tensor, Tensor]],
        torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
    )

    return train_loader, val_loader
