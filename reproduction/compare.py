"""Compare completed-epoch recovery snapshots after controlled GPU interruption."""
import argparse
import math

import numpy as np
import torch

from reproduction.checkpoint import load_checkpoint


def compare(left, right, rtol=1e-5, atol=1e-7, location='snapshot'):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol, msg=location)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right, err_msg=location)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'Different keys: {location}')
        for key in left:
            compare(left[key], right[key], rtol, atol, f'{location}.{key}')
    elif isinstance(left, (tuple, list)):
        if len(left) != len(right):
            raise AssertionError(f'Different lengths: {location}')
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, rtol, atol, f'{location}[{index}]')
    elif isinstance(left, float):
        if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f'Different values: {location}: {left} vs {right}')
    elif left != right:
        raise AssertionError(f'Different values: {location}: {left} vs {right}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uninterrupted')
    parser.add_argument('resumed')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-7)
    args = parser.parse_args()
    left, right = load_checkpoint(args.uninterrupted), load_checkpoint(args.resumed)
    for key in ['format_version', 'stage', 'epoch', 'config', 'best', 'model',
                'optimizer', 'lr_scheduler', 'loss_scaler']:
        compare(left[key], right[key], args.rtol, args.atol, key)
    compare(left['rng'], right['rng'], 0, 0, 'rng')
    print(f'Recovery state matches (rtol={args.rtol}, atol={args.atol}; RNG exact).')


if __name__ == '__main__':
    main()
