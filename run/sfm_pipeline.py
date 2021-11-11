import argparse
from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization

# Parser section
parser = argparse.ArgumentParser(
    description="Arguments for sparse SFM pipeline")
parser.add_argument("--dpath", type=str,
                    required=True,
                    help="Path to dataset wit images")
parser.add_argument("--opath", type=str, required=False)
parser.add_argument("--fmode", type=str, choices=["Superpoint"],
                    required=True,
                    help="Type of feature extraction")
parser.add_argument("--mmode", type=str, choices=["NN", "Superglue"],
                    required=False,
                    help="Type of matching, note that \
                                            Superglue only available with Superpoint")


def reconstruction(args):
    dataset = Path(args.dpath)
    images = dataset / "images"

    outputs = Path("outputs/sfm")
    sfm_pairs = outputs / "pairs-exhaustive.txt"  # exhaustive matching
    sfm_dir = outputs / "sfm_superpoint+superglue"

    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs, exhaustive=True)
    
    reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)

    pass


if __name__ == "__main__":
    args = parser.parse_args()
    reconstruction(args)
    pass
