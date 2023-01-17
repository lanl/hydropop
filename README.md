# hydropop
Creating scaling units to bridge ESM and "local" scale models

# Installation
Instructions here are provided for installation into an Anaconda environment on a Windows machine. I recommend using `mamba` as a drop-in replacement for most `conda` commands. It's a package-solver that's orders of magnitude faster. You can install it into your base environment with

`conda install mamba -c conda-forge`

or follow the actual [installation instructions](https://mamba.readthedocs.io/en/latest/installation.html).

Clone the `hydropop` repo to your PC. This will include the `hp_enviornment.yml` file containing the dependencies.

Open an Anaconda Terminal window and create an empty environment named `hprepo`:

`conda create --name hprepo --no-default-packages`

Then update this enviornment using `mamba` and pointing to the `hp_environment.yml` file:

`mamba env update -n hprepo --file "path\to\hp_environment.yml"`

Finally, use `pip` to do a live-install of the repo into the environment.

`conda activate hprepo`

`pip install -e "path\to\setup.py" # setup.py is in the hydropop repo you cloned`

And that's it! Note that this is a "live install" of the hydropop repo, which means that as you edit (or update) the repo's code on your PC, those changes will be immediately recognized.

# License
This repo is released with a [BSD-3 License](https://github.com/lanl/hydropop/blob/main/LICENSE).