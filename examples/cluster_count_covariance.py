#!/usr/bin/env python
# coding: utf-8

# ### Theoretical cluster count covariances using TJPCov.
# This script shows how to use TJPCov to calculate covariances, given a .sacc
# file.
#
# We will
# 1. Read in an appropriate yaml file that specifies to do cluster count
# covariances
# 2. Instantiate a `CovarianceCalculator`
# 3. Tell the calculator to calculate the covariances
# 4. Display the covariance / correlation coefficient
# 5. Save the covariance back to .sacc

# In[]:

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as t
import matplotlib.colors as colors
import sacc

from tjpcov.covariance_calculator import CovarianceCalculator

# In[]:

# The .yaml file contains the reference to the .sacc file - so specify which
# .sacc file you want to use there. You can also specify if you want to use
# MPI or not. Before running this example, run the notebook in the same
# folder to generate the sacc data.
input_yml = "./clusters/conf_covariance_clusters.yaml"
cc = CovarianceCalculator(input_yml)


# All we need to do is call `get_covariance` to calculate the covariance of
# all classes specified in the yaml file.  For this example we are calculating
# the cluster count covariance SSC and Gaussian contributions.
#
# Then, we call `create_sacc_cov` to the covariance to the sacc file.

# In[]:

cov = cc.get_covariance()
sacc_with_cov = cc.create_sacc_cov(
    output="mock_clusters_with_cov.sacc", save_terms=False
)


# Now we can either get the covariance matrix
#  from the covariance class directly, or from the SACC file (which we will
# do below).

# In[]:


cov_from_file = sacc_with_cov.covariance.covmat
# cov_from_file = sacc.Sacc.load_fits(
#   './clusters/mock_clusters_with_cov.sacc'
# ).covariance.covmat


# In[]:


def plot_cov_corr(cov, fig, ax):
    diag = np.diag(cov)
    corr = cov / np.sqrt(diag[:, None] * diag[None, :])

    im1 = ax[0].imshow(cov, cmap="Reds")
    im2 = ax[1].imshow(corr, cmap="bwr", vmax=1, vmin=-1)

    ax[0].set_title("Covariance")
    ax[1].set_title("Correlation")

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    for a in ax:
        a.set_yticklabels([])
        a.set_xticklabels([])
        a.set_xticks([])
        a.set_yticks([])
        a.yaxis.set_minor_locator(t.FixedLocator((8.75, 26.25)))
        a.yaxis.set_minor_formatter(
            t.FixedFormatter((r"$\mathcal{N}$", r"$\mathcal{M}$"))
        )
        a.xaxis.set_minor_locator(t.FixedLocator((8.75, 26.25)))
        a.xaxis.set_minor_formatter(
            t.FixedFormatter((r"$\mathcal{N}$", r"$\mathcal{M}$"))
        )


# Below we show the covariance and correlation

# In[]:


fig, ax = plt.subplots(1, 2, figsize=(8, 4))
plot_cov_corr(cov_from_file, fig, ax)


# We can also show the gaussian and SSC contributions to the covariance
# independently, shown below

# In[]:


cov_terms = cc.get_covariance_terms()


# In[]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
im1 = ax1.imshow(cov_terms["gauss"], norm=colors.LogNorm(), cmap="Reds")
im2 = ax2.imshow(cov_terms["SSC"], norm=colors.LogNorm(), cmap="Reds")
ax1.set_title("Shot Noise")
ax2.set_title("Sample Variance")

plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)

for a in [ax1, ax2]:
    a.set_yticklabels([])
    a.set_xticklabels([])
    a.set_xticks([])
    a.set_yticks([])
    a.yaxis.set_minor_locator(t.FixedLocator((8.75, 26.25)))
    a.yaxis.set_minor_formatter(
        t.FixedFormatter((r"$\mathcal{N}$", r"$\mathcal{M}$"))
    )
    a.xaxis.set_minor_locator(t.FixedLocator((8.75, 26.25)))
    a.xaxis.set_minor_formatter(
        t.FixedFormatter((r"$\mathcal{N}$", r"$\mathcal{M}$"))
    )

# %%
