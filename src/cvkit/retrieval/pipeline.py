"""RootSIFT + VLAD image-retrieval pipeline.

Implements the classic instance-retrieval recipe:

1. extract RootSIFT descriptors (Hellinger-normalised SIFT);
2. build a visual codebook with mini-batch k-means;
3. encode every image as a VLAD vector of residuals;
4. normalise with signed square-root and L2;
5. rank by cosine distance, optionally after exemplar-SVM re-encoding.

The module is importable and also exposes a CLI (``python -m cvkit
retrieval run``); the ablation drivers live in ``experiments/``.
"""

import os
import shlex
import argparse
from tqdm import tqdm

# Pickle helpers for gzip cache files; the original skeleton used the cPickle style.
import _pickle as cPickle

import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
import numpy as np
import cv2

def parseArgs(parser):
    parser.add_argument('--labels_test', 
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', 
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-str', '--suffix_train',
                        default='.png',
                        help='only chose those images with a specific suffix')
    parser.add_argument('-ste', '--suffix_test',
                        default='.jpg',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--to_binary', action='store_true',
                       help='use OTSU binarization')

    parser.add_argument('--in_test',
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train',
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    # Whether to apply the signed square-root power normalization common in VLAD.
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    # Compute base and powernorm VLAD in one pass, extracting test-image SIFT only once.
    parser.add_argument('--both', action='store_true',
                        help='compute and evaluate base VLAD and powernorm VLAD in one pass')
    # Number of visual words K. 100 is the standard VLAD setting and the default.
    parser.add_argument('--n_clusters', default=100, type=int,
                        help='number of visual words for the codebook')
    # Max local SIFT descriptors sampled from the training set for KMeans (about 500k).
    parser.add_argument('--max_descriptors', default=500000, type=int,
                        help='maximum number of SIFT descriptors sampled for KMeans')
    # To bound SIFT extraction time before KMeans, sample only 100 training images by default.
    parser.add_argument('--max_dictionary_files', default=100, type=int,
                        help='number of training files sampled for KMeans; use 0 for all files')
    # E-SVM is slow, so it is off by default; pass --esvm explicitly to run it.
    parser.add_argument('--esvm', action='store_true',
                        help='also run exemplar SVM after VLAD evaluation')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    return parser

def getFiles(folder, pattern, labelfile):
    """ 
    returns files and associated labels by reading the labelfile 
    parameters:
        folder: inputfolder
        pattern: new suffix
        labelfiles: contains a list of filename and labels
    return: absolute filenames + labels 
    """
    # Read label file. Each line format is: image_name author/class_id.
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    
    # Build the real image paths from the label file while keeping each writer label.
    all_files = []
    labels = []
    for line in all_lines:
        # shlex can handle filenames with quotes/escaped spaces.
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]

        # Strip any suffix already present in the label file, then append the dataset suffix.
        # We don't use os.path.splitext here because some filenames may contain '.' in the middle.
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p,'')

        # Concatenate the full path, e.g., training_folder + image_id + .png.
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)

    return all_files, labels

def loadRandomDescriptors(files, max_descriptors, max_files=100):
    """ 
    load roughly `max_descriptors` random descriptors
    parameters:
        files: list of filenames containing local features of dimension D
        max_descriptors: maximum number of descriptors (Q)
    returns: QxD matrix of descriptors
    """
    # Training the codebook only needs a random subset of images and descriptors.
    files = np.array(files)
    if max_files is not None and max_files > 0:
        max_files = min(max_files, len(files))
        indices = np.random.permutation(len(files))[:max_files]
        files = files[indices]
    else:
        # With max_files<=0, use all training images but still shuffle to avoid label-order bias.
        files = files[np.random.permutation(len(files))]
   
    # Cap the SIFT count per image so a few large images cannot dominate KMeans.
    max_descs_per_file = max(1, int(np.ceil(max_descriptors / len(files))))

    descriptors = []
    for i in tqdm(range(len(files)), desc='sample SIFT'):
        # Use RootSIFT: extract SIFT first, then apply Hellinger normalization.
        desc = computeDescs(files[i], True, True)
        if len(desc) == 0:
            continue
            
        # Randomly sample from all SIFT descriptors of the current image.
        indices = np.random.choice(len(desc),
                                   min(len(desc),
                                       int(max_descs_per_file)),
                                   replace=False)
        desc = desc[ indices ]
        descriptors.append(desc)
    
    if len(descriptors) == 0:
        raise RuntimeError('no descriptors extracted from the selected files')

    # Concatenate into a Qx128 matrix as input to MiniBatchKMeans.
    descriptors = np.concatenate(descriptors, axis=0).astype(np.float32)
    if len(descriptors) > max_descriptors:
        # If the total exceeds the limit due to ceil, perform one more global random truncation.
        indices = np.random.choice(len(descriptors), max_descriptors, replace=False)
        descriptors = descriptors[indices]
    return descriptors

def toBinary(mask):
    if mask is None:
        raise ValueError('cannot binarize an empty image')

    # If the image is already a 0/255 binary mask, return it; otherwise apply Otsu.
    if mask[mask==255].sum() != np.sum(mask):
        # Some binary images may be 0/1 and need to be converted to OpenCV's standard 0/255.
        if mask[mask==1].sum() == mask.sum():
            mask *= 255
        else: # For regular grayscale images: use Otsu to find the threshold.
           ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return mask

def computeDescs(fname, norm_hellinger=False, to_binary=False):
    # Read the handwritten image in grayscale; SIFT only requires single-channel images.
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(fname)

    if to_binary:
        # The data is already binary; keep the switch so the input stays uniformly binary.
        img = toBinary(img)

    # OpenCV SIFT: detect keypoints, then compute 128-D descriptors at them.
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)
    for keypoint in keypoints:
        # Keypoint orientation is fixed to 0, so descriptors are rotation-sensitive.
        keypoint.angle = 0.0

    keypoints, descriptors = sift.compute(img, keypoints)
    if descriptors is None:
        # If an image has no keypoints, return an empty 0x128 matrix; VLAD then becomes zero.
        return np.empty((0, 128), dtype=np.float32)

    descriptors = descriptors.astype(np.float32)
    if norm_hellinger:
        # Hellinger/RootSIFT normalization:
        # L1-normalize then sqrt each dimension; better than raw SIFT for Euclidean/KMeans.
        eps = np.finfo(np.float32).eps
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + eps)
        descriptors = np.sqrt(descriptors)
    
    return descriptors

def dictionary(descriptors, n_clusters):
    """ 
    return cluster centers for the descriptors 
    parameters:
        descriptors: NxD matrix of local descriptors
        n_clusters: number of clusters = K
    returns: KxD matrix of K clusters
    """
    # Train the visual word codebook using SIFT descriptors sampled from the training set.
    # MiniBatchKMeans is much faster than standard KMeans and is suitable for this scale of ~500k.
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=max(1000, n_clusters * 20),
        n_init=3,
        random_state=42,
        verbose=0,
    )
    kmeans.fit(descriptors)
    # The shape of cluster_centers_ is Kx128, i.e., K visual word centers.
    return kmeans.cluster_centers_.astype(np.float32)

def nearest_cluster_indices(descriptors, clusters):
    """Return the nearest visual word index for each local descriptor."""
    descriptors = descriptors.astype(np.float32, copy=False)
    clusters = clusters.astype(np.float32, copy=False)

    # Calculate distance using ||x-c||^2 = ||x||^2 + ||c||^2 - 2*x*c.
    # This avoids explicitly creating a 3D residual tensor of T x K x D, making memory more stable.
    desc_norms = np.sum(descriptors * descriptors, axis=1, keepdims=True)
    cluster_norms = np.sum(clusters * clusters, axis=1)
    dists = desc_norms + cluster_norms[None, :] - 2.0 * descriptors.dot(clusters.T)
    return np.argmin(dists, axis=1)

def assignments(descriptors, clusters):
    """ 
    compute assignment matrix
    parameters:
        descriptors: TxD descriptor matrix
        clusters: KxD cluster matrix
    returns: TxK assignment matrix
    """
    if len(descriptors) == 0:
        return np.empty((0, len(clusters)), dtype=np.float32)

    # Find the nearest cluster center for each SIFT descriptor.
    nearest = nearest_cluster_indices(descriptors, clusters)

    # Hard assignment: each descriptor goes to its nearest word, encoded as one-hot.
    assignment = np.zeros( (len(descriptors), len(clusters)), dtype=np.float32 )
    assignment[np.arange(len(descriptors)), nearest] = 1.0

    return assignment

def raw_vlad(desc, mus):
    K, D = mus.shape
    if len(desc) == 0:
        # Without SIFT, cannot accumulate residuals; return an all-zero VLAD vector.
        return np.zeros((D * K), dtype=np.float32)

    # VLAD core: assign each descriptor to its nearest word and accumulate residuals.
    nearest = nearest_cluster_indices(desc, mus)
    residuals = desc - mus[nearest]
    f_enc = np.zeros((K, D), dtype=np.float32)
    # np.add.at adds all residuals of the same cluster to the corresponding row.
    np.add.at(f_enc, nearest, residuals)
    # Flatten KxD into K*D: a 12800-D image feature when K=100 and D=128.
    return f_enc.ravel()

def normalize_vlad(f_enc, powernorm):
    f_enc = f_enc.astype(np.float32, copy=True)
    if powernorm:
        # Signed square-root: damp very large residual dimensions and limit burstiness.
        f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

    # L2-normalize so cosine similarity reduces to a plain dot product.
    norm = np.linalg.norm(f_enc)
    if norm > 0:
        f_enc /= norm
    return f_enc.astype(np.float32)

def vlad(files, mus, powernorm):
    """
    compute VLAD encoding for each files
    parameters: 
        files: list of N files containing each T local descriptors of dimension
        D
        mus: KxD matrix of cluster centers
        gmp: if set to True use generalized max pooling instead of sum pooling
    returns: NxK*D matrix of encodings
    """
    encodings = []

    for f in tqdm(files, desc='VLAD'):
        # Turn each image into a variable number of RootSIFT vectors, then pool to VLAD.
        desc = computeDescs(f, True, True)
        f_enc = raw_vlad(desc, mus)
        encodings.append(normalize_vlad(f_enc, powernorm))

    # Return an Nx(K*D) matrix, where N is the number of images.
    return np.vstack(encodings).astype(np.float32)

def vlad_both(files, mus):
    # Compute each image's raw VLAD once, then derive base and powernorm versions.
    # This avoids re-extracting SIFT in --both mode, saving a lot of time.
    enc_base = []
    enc_power = []

    for f in tqdm(files, desc='VLAD'):
        desc = computeDescs(f, True, True)
        f_enc = raw_vlad(desc, mus)
        enc_base.append(normalize_vlad(f_enc, False))
        enc_power.append(normalize_vlad(f_enc, True))

    return (
        np.vstack(enc_base).astype(np.float32),
        np.vstack(enc_power).astype(np.float32),
    )

def esvm(encs_test, encs_train, C=1000):
    """ 
    compute a new embedding using Exemplar Classification
    compute for each encs_test encoding an E-SVM using the
    encs_train as negatives   
    parameters: 
        encs_test: NxD matrix
        encs_train: MxD matrix

    returns: new encs_test matrix (NxD)
    """
    # Train a binary classifier for each test sample:
    # Positive sample is the current test image, negative samples are all training images.
    y = np.hstack(([1], -np.ones(len(encs_train), dtype=np.int32)))

    def loop(i):
        # Train an exemplar SVM for the current test image.
        x = np.vstack((encs_test[i:i+1], encs_train))
        clf = LinearSVC(C=C, class_weight='balanced', max_iter=10000)
        clf.fit(x, y)
        # Use the SVM's weight vector coef_ as the new embedding for this test image.
        transformed = clf.coef_.astype(np.float32)
        norm = np.linalg.norm(transformed)
        if norm > 0:
            transformed /= norm
        return transformed

    # One SVM is trained per query, which is the dominant cost of E-SVM.
    new_encs = list(map(loop, tqdm(range(len(encs_test)), desc='E-SVM')))
    new_encs = np.concatenate(new_encs, axis=0)
    # Return the E-SVM embeddings for all test images.
    return new_encs


def distances(encs):
    """ 
    compute pairwise distances 

    parameters:
        encs:  TxK*D encoding matrix
    returns: TxT distance matrix
    """
    # First, to be safe, perform L2 normalization again, then use 1-dot as cosine distance.
    encs = encs.astype(np.float32, copy=False)
    norms = np.linalg.norm(encs, axis=1, keepdims=True)
    encs = encs / np.maximum(norms, np.finfo(np.float32).eps)
    dists = 1.0 - encs.dot(encs.T)
    # Set self-distances to the maximum so a query never retrieves itself first.
    np.fill_diagonal(dists, np.finfo(dists.dtype).max)
    return dists

def evaluate(encs, labels):
    """
    evaluate encodings assuming using associated labels
    parameters:
        encs: TxK*D encoding matrix
        labels: array/list of T labels
    """
    dist_matrix = distances(encs)
    # Each row corresponds to a query; sort database images by distance in ascending order.
    indices = dist_matrix.argsort()

    n_encs = len(encs)

    mAP = []
    correct = 0
    for r in range(n_encs):
        # Relevant images for query r are images whose label is the same as the query.
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[ indices[r,k] ] == labels[ r ]:
                rel += 1
                # Each time a relevant image is found, record the precision at the current position.
                precisions.append( rel / float(k+1) )
                if k == 0:
                    correct += 1
        # The AP for the current query; the average AP of all queries is the mAP.
        avg_precision = np.mean(precisions) if len(precisions) > 0 else 0.0
        mAP.append(avg_precision)
    mAP = np.mean(mAP)

    top1 = float(correct) / n_encs
    print('Top-1 accuracy: {} - mAP: {}'.format(top1, mAP))
    return top1, mAP


def main(argv=None):
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args(argv)
    # Fix the random seed to ensure descriptor sampling and KMeans initialization are reproducible.
    np.random.seed(42)

    # These path arguments are required; without them data or labels cannot be located.
    for required_arg in ['in_train', 'in_test', 'labels_train', 'labels_test']:
        if getattr(args, required_arg) is None:
            raise ValueError('--{} is required'.format(required_arg))

    # a) Read the training set file list and build a codebook using training set SIFT descriptors.
    files_train, labels_train = getFiles(args.in_train, args.suffix_train,
                                         args.labels_train)
    assert (len(files_train) == len(labels_train))
    print('#train: {}'.format(len(files_train)))

    missing_train = [f for f in files_train if not os.path.exists(f)]
    if missing_train:
        raise FileNotFoundError('missing training file, e.g. {}'.format(missing_train[0]))

    # The codebook cache name encodes key parameters, so stale codebooks are not reused.
    dict_fname = 'mus_k{}_d{}_f{}.pkl.gz'.format(
        args.n_clusters, args.max_descriptors, args.max_dictionary_files)

    if not os.path.exists(dict_fname) or args.overwrite:
        # On first run: randomly sample RootSIFT from training images, then train a KMeans codebook.
        descriptors = loadRandomDescriptors(
            files_train,
            args.max_descriptors,
            args.max_dictionary_files,
        )
        print('> computed/loaded {} descriptors'.format(len(descriptors)))

        # KMeans cluster centers are the visual word codebook for VLAD.
        print('> compute dictionary')
        mus = dictionary(descriptors, args.n_clusters)
        with gzip.open(dict_fname, 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        # For subsequent experiments, load the codebook directly to save KMeans time.
        print('> load dictionary from {}'.format(dict_fname))
        with gzip.open(dict_fname, 'rb') as f:
            mus = cPickle.load(f)


    # b/c/d) Perform VLAD encoding on test set images and evaluate retrieval.
    print('> compute VLAD for test')
    files_test, labels_test = getFiles(args.in_test, args.suffix_test,
                                       args.labels_test)
    print('#test: {}'.format(len(files_test)))

    missing_test = [f for f in files_test if not os.path.exists(f)]
    if missing_test:
        raise FileNotFoundError('missing test file, e.g. {}'.format(missing_test[0]))

    # Cache base VLAD and powernorm VLAD separately, with K recorded in the filename.
    base_fname = 'enc_test_k{}_base.pkl.gz'.format(args.n_clusters)
    power_fname = 'enc_test_k{}_powernorm.pkl.gz'.format(args.n_clusters)

    if args.both:
        # --both: read the images once and build both normalized VLAD variants.
        if (not os.path.exists(base_fname) or not os.path.exists(power_fname)
                or args.overwrite):
            enc_base, enc_power = vlad_both(files_test, mus)
            with gzip.open(base_fname, 'wb') as fOut:
                cPickle.dump(enc_base, fOut, -1)
            with gzip.open(power_fname, 'wb') as fOut:
                cPickle.dump(enc_power, fOut, -1)
        else:
            print('> load test encodings from {} and {}'.format(base_fname, power_fname))
            with gzip.open(base_fname, 'rb') as f:
                enc_base = cPickle.load(f)
            with gzip.open(power_fname, 'rb') as f:
                enc_power = cPickle.load(f)

        # Evaluate base and power-normalised VLAD separately for side-by-side comparison.
        print('> evaluate base VLAD')
        evaluate(enc_base, labels_test)
        print('> evaluate VLAD + powernorm')
        evaluate(enc_power, labels_test)
        raise SystemExit(0)

    suffix = 'powernorm' if args.powernorm else 'base'
    fname = power_fname if args.powernorm else base_fname
    if not os.path.exists(fname) or args.overwrite:
        # Standalone mode: compute only the user-specified base or powernorm version.
        enc_test = vlad(files_test, mus, args.powernorm)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        # If cache exists, load directly and evaluate.
        print('> load test encodings from {}'.format(fname))
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)

    # Retrieval evaluation inside the test set: each test image queries all the others.
    print('> evaluate {}'.format(suffix))
    evaluate(enc_test, labels_test)

    if args.esvm:
        # Optional e) Exemplar SVM: requires training set VLAD as negative samples.
        print('> compute VLAD for train (for E-SVM)')
        fname = 'enc_train_k{}_{}.pkl.gz'.format(args.n_clusters, suffix)
        if not os.path.exists(fname) or args.overwrite:
            enc_train = vlad(files_train, mus, args.powernorm)
            with gzip.open(fname, 'wb') as fOut:
                cPickle.dump(enc_train, fOut, -1)
        else:
            print('> load train encodings from {}'.format(fname))
            with gzip.open(fname, 'rb') as f:
                enc_train = cPickle.load(f)

        # Replace the original test embedding with E-SVM weights and re-evaluate retrieval.
        print('> esvm computation')
        enc_test = esvm(enc_test, enc_train, args.C)

        print('> evaluate E-SVM')
        evaluate(enc_test, labels_test)


if __name__ == '__main__':
    main()
