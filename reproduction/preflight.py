import argparse
import json
from reproduction.data import validate_voc
from reproduction.records import transfer_manifest, write_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-root')
    parser.add_argument('--pseudo-dir')
    parser.add_argument('--train-list', default='voc12/train_aug_id.txt')
    parser.add_argument('--val-list', default='voc12/val_id.txt')
    parser.add_argument('--record', default='artifacts/preflight.json')
    parser.add_argument('--transfer-dir')
    parser.add_argument('--verify-transfer', action='store_true')
    args = parser.parse_args()
    if args.transfer_dir:
        print(json.dumps(transfer_manifest(args.transfer_dir, args.verify_transfer), indent=2))
    else:
        if not args.voc_root:
            parser.error('--voc-root is required for dataset preflight')
        result = validate_voc(args.voc_root, args.train_list, args.val_list, args.pseudo_dir)
        write_record(args.record, dict(arguments=vars(args), validation=result))
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
