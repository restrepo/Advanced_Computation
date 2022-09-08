#!/usr/bin/env python
# coding: utf-8

# # GitHub Action for DevOps 
# The Development and IT Operations (DevOps) implemented here trough GitHub actions includes three parts
# 1. Python style guide enforcement through `flake8`. Check the `jupyterlab-flake8` extension or the `cornflakes-linter` extension of visual code, to easily fix possible syntax errors. They are activated when the file is saved. The corresponding GitHub action is triggered after each _commit_.
# 2. Testing through `pytest`. The corresponding GitHub action is triggered after each _commit_.
# 3. Creating a pip package and publish it in (test) pypi.  The corresponding GitHub action is triggered after each _release_.

# ## Simple template
# To illustrate this three points we will use a template repository in GitHub with a very simple Python program that just returns the string: `'Hello, World!'`:
# 
# https://github.com/restrepo/devops
# 
# From there we will create a pip package with a modified version of the template in the sandbox of https://pypi.org, which is called https://test.pypi.org. An official package could follow the same procedure but by using the official site. Please make a fork of the previous repository and rename it to some name that is not used at all in https://test.pypi.org, for example, for `abc123`
# 
# https://test.pypi.org/project/abc123/
# 
# the message: 
# > We looked everywhere but couldn't find this page
# 
# would appears there. This means that the name will be avalaible for a new (test) pip package under the name `abc123`. I will use that name from now on
# 
# ### (Test) pypi configuration
# Please creates an account in https://test.pypi.org, login there, and go to the ["Account Settings"](https://test.pypi.org/manage/account/) (from the uper-right link with your chosen username).
# 
# After the proper authentication, scroll down to the section [API tokens](https://test.pypi.org/manage/account/#api-tokens) and click on the button:
# <button style="background-color:#006dad; border-color:#006dad; color: #fff;">Add API token</button>. Fill the fields "Token name", with `github` for example, and the scope with `Entire account (all projects)`, after add the token, you must copy it in your clipboard this but using the button: <button style="background-color:#006dad; border-color:#006dad; color: #fff;">Copy token</button>.

# ### GitHub settings
# In the GitHub page for your repository, that from now on I will use as (please use your own one from now on)
# 
# https://github.com/MyAccountInGitHub/abc123
# 
# Go to the Settings tab and from there follow the following navigation:
# * Secrets → Actions → Actions secrets → <button> New repository secret</button> and fill the fields with:
#   * __Name__: TESTPYPI_PASSWORD
#   * __Secret__: Paste your copied token here
# 
# Be sure that after add the Secret, a confirmation message appears in the top of the page with:
# > Repository secret added.
#   
# Go back  to the GitHub page for your repository at: 
# 
# https://github.com/MyAccountInGitHub/abc123
# 
# Go to the Actions tab and if necessary, click on: 
# 
# <button style="background-color:#2da44e; border-color:rgba(27,31,36,0.15); color: #fff;">I understand my workflows, go ahead and enable them</button>

# ### GitHub Actions
# #### Python package action
# In `.github/workflows/python-package.yml` there are the instructions for the creation of [Ubuntu](https://ubuntu.com/) [Docker containers](https://en.wikipedia.org/wiki/Docker_(software)) with the clonned repository, specific python versions, and the required dependencies (see below). Also, there are scripts to run the commands: `flake8` and `pytest` upon the repository. Please check the file for the details.
# 
# This processes is triggered after each commit. Please edit the `setup.py`  and replace the following lines there
# ```python
#         # Application name:
#         name="desoper",
# 
#         # Version number (initial):
#         version="0.0.3",
# ```
# by the new repository name, e,g: `abc123`, and reset the version to 0.0.1:
# ```python
#         # Application name:
#         name="abc123",
# 
#         # Version number (initial):
#         version="0.0.1",
# ```
# Then, make the commit and check again the Actions tab. Wait until all the processes end up in green and check the Output by clicking in the link with the commit name, e.g "Update setup.py".  From there click in the link with the last build preceded by the check green circle, e.g build (3.9). In the log output, expand the arrow with `Lint with flake8`  and the arrow with `Test with pytest`.
# 
# Also make the following corresponding changes and edits (use your own package name instead of `abc123`) , because the pypi server check for files with the same name or contents:
# 
# 1. Inside the directory `desoper` change the name of the file `hello.py`  to `abc123.py`
# 1. Edit `abc123.py` and change in line 3 `Hello module` by  `abc123 module`
# 1. Edit `__init__.py ` and change:
# ```python
# from desoper import hello
# ```
# by
# ```python
# from desoper import abc123
# ```

# #### Python publish action
# In `.github/workflows/python-publish.yml` the deploy processes in (test.)pypi.org are automated by creating the pip package and uploading it to the (test) pypi official server. 
# This action is trigger when a new release is created. For that, go again to the main page of the repository and click on the link of the left called [__Releases__](https://github.com/MyAccountInGitHub/abc123/releases). Once there, 
# 1. Click in the button <button style="background-color:#2da44e; border-color:rgba(27,31,36,0.15); color: #fff;">Create a new release</button>
# 1. Click in the menu "Choose a tag" and in fill the field with "v0.0.1" and click in the button <button>__+ Create new tag: v0.0.1__ on publish</button>
# 1. Fill the "Release title" box with some description, e.g., `Initial release`.
# 1. Click the button <button style="background-color:#2da44e; border-color:rgba(27,31,36,0.15); color: #fff;">Publish release</button>
# 1. Check again the Actions tab and wait for the deploy workflow to finish, and check that it finished in green 
# 1. Check that the tags in the the `README.md` are updated with a green status (in an incognito window to avoid the browser cache)
# 1. Go to the https://test.pypi.org/project/abc123/ link again and check for the installation instructions
# 
# Try to install the package in your local computer
# ```bash
# $ pip install -i https://test.pypi.org/simple/ abc123
# ```
# Note that the name of the package [is different](https://stackoverflow.com/a/53346523/2268280) from the _name of the module_, given by the name of the main directory inside the repository. Therefore, to use the module, we need to take into account the corresponding path in the repository:
# 
# | package name | main directory | module file | function |
# |--------------|----------------|-------------|----------|
# | `abc123`     | `desoper`      | `abc123.py` | `hello()`|
# 
# ```python
# $ python3
# >>> import desoper
# >>> desoper.abc123.hello()
# 'Hello, World!
# ```
# 
# 
# 
# Note that for new releases,  the `setup.py` file must be updated with the proper release number, e.g: `0.0.2`, and the related commit must pass all the previous tests.
# 

# __Activity__: Update the README.md for the new package name and the new module file name, make a new release and upgrade with local pip installation (Hint: Use the `-U` option at the end of the pip install command)

# __Activity__: Uninstall the package in your local computer

# For an example with dependencies management inside the `setup.py` file, check the `anomalies` package at https://github.com/restrepo/anomalies, where specific `numpy` version are required:
# > See: https://docs.scipy.org/doc/scipy/dev/toolchain.html#numpy
# ```python
#         install_requires=[
#             'numpy==1.16.5; python_version=="3.7"',
#             'numpy>=1.16.5; python_version=="3.8"',
#             'numpy>=1.16.5; python_version=="3.9"'
#         ],
# ```
# 

# ## Local management of DevOps
# 
# ### Install
# For that, we must can be clone the corresponding repository in your local computer in some directory at a second level from `$HOME`:
# ```bash
# mkdir -p prog
# cd prog
# #Clone the repo in $HOME/prog
# git clone https://github.com/restrepo/anomalies.git
# ```
# From here, be sure that you don't have rhe package already installed
# ```
# pip3 uninstall anomalies
# ```
# with `sudo` if necessary.
# 
# The `setup.py` file in the main directory of the package contains the metadata and requirements of the package. This allow for the installation either at the user-level or the system-level. At the user level it is just
# ```bash
# python3 setup.py install --user
# ```
# 
# We can check the installation by returning back to the `$HOME` directory and loading the module from there
# 
# ```python
# cd
# python3
# >>> import anomalies
# >>> exit()
# ```
# 
# ### Syntax checking with `flake8`
# By returning back to the directory with the repository
# ```bash
# cd prog/anomalies
# ```
# Stop if Python syntax errors or undefined names:
# ```
# flake8 . --ignore=C901 --select=E9,F63,F7,F82
# ```
# Warning otherwise:
# ```
# flake8 . --count --ignore=C901 --exit-zero --max-complexity=10 --max-line-length=127 --statistics
# ```
# For further info in each rule, check the corresponding URL, e.g, for `E741`:
# 
# https://www.flake8rules.com/rules/E741.html
# 
# To easily fix the syntax errors you can use visual code with the extension `cornflakes-linter`. Whenever you save the file, a highlighter will point out to `flask8` syntax error messages.
# 
# ### Tests
# At the first level of the repository, use
# ```
# pytest
# ```

# ## References
# [1] https://en.wikipedia.org/wiki/DevOps
# 
# [2] https://dev.to/arnu515/create-a-pypi-pip-package-test-it-and-publish-it-using-github-actions-part-2-1o83

# In[ ]:




