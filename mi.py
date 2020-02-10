from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib


plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'


def get_img_slice(img, verbose=False):
    """
    load the image using nibabel and slice the image

    Parameters
    ----------
    img: nii image data read via nibabel

    Returns
    -------
    numpy memmap: ndarray of image slice
    """
    img_data = nib.load(img)
    img_data = img_data.get_data()

    # get the middle slice
    assert len(np.shape(img_data)) == 3
    z_slice = int(np.shape(img_data)[-1]/2)
    img_slice = img_data[:, :, z_slice]

    # convert any nans to 0s
    img_nans = np.isnan(img_slice)
    img_slice[img_nans] = 0
    if verbose:
        print('slice index: ' + str(z_slice))
    return img_slice

# example
# img1_slice = get_img_slice(img1)
# img2_slice = get_img_slice(img2)


# display images left and right
def plot_raw(img1, img2):
    plt.imshow(np.hstack((img1, img2)))
    plt.show()

# example
# plot_raw(img1_slice, img2_slice)


def plot_hist1d(img1, img2, bins=20):
    """
    one dimensional histogram of the slices

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram

    Returns
    -------
    histogram
        comparing two images side by side
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(img1.ravel(), bins)
    axes[0].set_title('Img1 histogram')
    axes[1].hist(img2.ravel(), bins)
    axes[1].set_title('Img2 histogram')
    plt.show()

# example
# plot_hist1d(img1_slice, img2_slice)


def plot_scatter2d(img1, img2):
    """
    plot the two image's histogram against each other

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    Returns
    -------
    2d plotting of the two images
    """
    corr = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
    plt.plot(img1.ravel(), img2.ravel(), '.')

    plt.xlabel('Img1 signal')
    plt.ylabel('Img2 signal')
    plt.title('Img1 vs Img2 signal cc=' + str(corr))
    plt.show()

# example
# plot_scatter2d(img1_slice, img1_slice)
# plot_scatter2d(img1_slice, img2_slice)


def plot_joint_histogram(img1, img2, bins=20, log=True):
    """
    plot feature space. Given two images, the feature space is constructed by counting the number of times a combination of grey values occur

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram
    log: boolean (default=True)
        keeping it true will show a better contrasted image

    Returns
    -------
    joint histogram
        feature space of the two images in graph

    """
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins)
    # transpose to put the T1 bins on the horizontal axis and use 'lower' to put 0, 0 at the bottom of the plot
    if not log:
        plt.imshow(hist_2d.T, origin='lower')
        plt.xlabel('Img1 signal bin')
        plt.ylabel('Img2 signal bin')

    # log the values to reduce the bins with large values
    hist_2d_log = np.zeros(hist_2d.shape)
    non_zeros = hist_2d != 0
    hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
    plt.imshow(hist_2d_log.T, origin='lower')
    plt.xlabel('Img1 signal bin')
    plt.ylabel('Img2 signal bin')

# example
# this should print a linear graph as it's comparing it to itself
# print(plot_joint_histogram(img1_slice, img1_slice))
# print(plot_joint_histogram(img1_slice, img2_slice))


def mutual_information(img1, img2, bins=20):
    """
    measure the mutual information of the given two images

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    bins: optional (default=20)
        bin size of the histogram

    Returns
    -------
    calculated mutual information: float

    """
    hist_2d, x_edges, y_edges = np.histogram2d(img1.ravel(), img2.ravel(), bins)

    # convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal x over y
    py = np.sum(pxy, axis=0)  # marginal y over x
    px_py = px[:, None] * py[None, :]  # broadcast to multiply marginals

    # now we can do the calculation using the pxy, px_py 2D arrays
    nonzeros = pxy > 0  # filer out the zero values
    return np.sum(pxy[nonzeros] * np.log(pxy[nonzeros] / px_py[nonzeros]))

# example
# the MI value of the first should be greater than the second as the first is comparing the image to itself
# print(mutual_information(img1_slice, img1_slice))
# print(mutual_information(img1_slice, img2_slice))


def get_core(img1, img2):
    """
    calculate correlation coefficient for given two images

    Parameters
    ----------
    img1: nii image data read via nibabel

    img2: nii image data read via nibabel

    Returns
    -------
    corr: float
        correlation coefficient value
    """
    corr = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
    return corr
