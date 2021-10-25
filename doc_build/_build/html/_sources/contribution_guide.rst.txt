.. _contributing_to_pyleoclim:

#########################
Contributing to Pyleoclim
#########################


GitHub, GitHub, GitHub,
=======================
All contributions, bug reports, bug fixes, documentation improvements, enhancements,
and ideas are welcome, and take place through [GitHub](https://github.com/LinkedEarth/Pyleoclim_util/issues).

There are several levels of contributions to an open development software package like Pyleoclim, including:

#. Reporting Bugs
#. Updating the documentation
#. Updating existing functionalities
#. Contributing new functionalities

All of that takes place through GitHub [issues](https://docs.github.com/en/issues/tracking*your*work*with*issues/quickstart), but we recommend first reaching out to our community Slack to avoid effort duplication (to get added to our Slack workspace, please email linkedearth@gmail.com).

When you start working on an issue, it’s a good idea to assign the issue to yourself, again to limit duplication. If you can’t think of an issue of your own, we have you convered:  check the list of unassigned issues and assign yourself one you like.
If for whatever reason you are not able to continue working with the issue, please try to unassign it, so other people know it’s available again. You can check the list of assigned issues, since people may not be working in them anymore. If you want to work on one that is assigned, feel free to kindly ask the current assignee (on GitHub and/or Slack) if you can take it (please allow at least a week of inactivity before considering work in the issue discontinued).

Bug reports and enhancement requests
====================================

Bug reports are an important part of improving any software. Having a complete bug report will allow others to reproduce the bug and provide insight into fixing. See this [stackoverflow article](https://stackoverflow.com/help/mcve) and [this blogpost](https://matthewrocklin.com/blog/work/2018/02/28/minimal*bug*reports) for tips on writing a good bug report.
Trying the bug*producing code out on the master branch is often a worthwhile exercise to confirm the bug still exists. It is also worth searching existing bug reports and pull requests to see if the issue has already been reported and/or fixed.
Bug reports must:

1. Include a minimal working example (a short, self*contained Python snippet reproducing the problem). You can format the code nicely by using GitHub Flavored Markdown:

.. code*block:: python

    import pyleoclim as pyleo
    url = 'https://raw.githubusercontent.com/LinkedEarth/Pyleoclim_util/Development/example_data/lipds.json'
    response = urlopen(url)
    D = json.loads(response.read())
    d=pyleo.Lipd(lipd_dict=D)
    ts = d.to_LipdSeries(number=6)
    res=ts.mapNearRecord(d,mute=True)

2. Include the full version string of pyleoclim, which you can obtain through:

.. code*block:: python

    pyleo.__version__

3. Explain why the current behavior is wrong/not desired and what you expect or would like to see instead.



Working with the Pyleoclim codebase
====================================
Version control, Git, and GitHub
""""""""""""""""""""""""""""""""

To the neophyte, working with Git is one of the more daunting aspects of contributing to open source projects.
It can very quickly become overwhelming, but sticking to the guidelines below will help keep the process straightforward and mostly trouble free. As always, if you are having difficulties please feel free to ask for help.
The code is hosted on [GitHub](https://github.com/LinkedEarth/Pyleoclim_util). To contribute you will need to [sign up for a (free) GitHub account](https://github.com/signup/free). [Git](https://git*scm.com/) is the industry standard for version control to allow many people to work together on the project, keep track of issues, manage the project, and much more.

Some great resources for learning Git:
  * the [GitHub help pages](https://help.github.com/).
  * the [NumPy documentation](https://numpy.org/doc/stable/dev/index.html).
  * Matthew Brett’s [Pydagogue](https://matthew*brett.github.io/pydagogue/).

GitHub has [instructions](https://help.github.com/set*up*git*redirect) for installing git, setting up your SSH key, and configuring git. All these steps need to be completed before you can work seamlessly between your local repository and GitHub.

Forking
"""""""
You will need your own fork to work on the code. Go to the Pyleoclim repository and hit the Fork button. You will then want to “clone” your fork (i.e. download all the code) to your local machine so you can edit it. At the command line, this would like something like:

.. code*block:: bash

    git clone https://github.com/your*user*name/Pyleoclim_util.git pyleoclim*yourname
    cd pyleoclim*yourname
    git remote add upstream https://github.com/LinkedEarth/Pyleoclim_util.git

This creates the directory _pyleoclim*yourname_ and connects your repository to the upstream (main project) Pyleoclim repository.  However, most Git first*timers may find it easier to do so through the Github web interface or desktop app (where there is a proverbial “button for that”).

Creating a development environment
""""""""""""""""""""""""""""""""""
We recommend developing in the same conda environment in which you installed pyleoclim (see :ref: `installation`).

Creating a branch
"""""""""""""""""
You want your master branch to reflect only production*ready code, so create a feature branch for making your changes. For example:

.. code*block:: bash

    git branch shiny*new*feature
    git checkout shiny*new*feature

The above can be simplified to:

:command: `git checkout *b shiny*new*feature`

This changes your working directory to the *shiny*new*feature* branch. Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to Pyleoclim. You can have many `shiny*new*features` and switch in between them using the :command: `git checkout` command.
When creating this branch, make sure your master branch is up to date with the latest upstream master version. To update your local master branch, you can do:

.. code*block:: bash

    git checkout master
    git pull upstream master **ff*only

When you want to update the feature branch with changes in master after you created the branch, check the section on updating a pull request.

Pyleoclim Protocol
""""""""""""""""""

Contributing new functionalities
********************************

  1. Open an issue on GitHub (See above)
  2. Implement outside of Pyleoclim
  Before incorporating any code into Pyleoclim, make sure you have a solution that works outside Pyleoclim. Demonstrate this in a notebook, which can be hosted on GitHub as well so it is easy for the maintainers to check out. The notebook should be organized as follows:
    * dependencies (package names and versions),
    * body of the function
    * example usage
  3. Integrate the new functionality
  Now you may implement the new functionality inside Pyleoclim. In so doing, make sure you:
    * Re*use as many of Pyleoclim’s existing utilities as you can, introducing new package  dependencies only as necessary.
    * Create a docstring for your new function, describing arguments and returned variables, and showing an example of use. (Use an existing docstring for inspiration).
    * If possible, also include a unit test for [continuous integration](https://youtu.be/_WvjhrZR01U) (Pyleoclim uses pytest and TravisCI). Feel free to ask for help from the package developers.
  4. Expose the new functionality in the Pyleoclim API (ui.py)


Updating existing functionalities
**********************************

1. Open an issue on GitHub (same advice as above)
2. Implement outside of Pyleoclim, including a benchmark of how the existing function performs vs the proposed upgrade (e.g. with :command: `timeit`).  Take into consideration memory requirements and describe on what architecture/OS you ran the test.
3. Integrate the new functionality within Pyleoclim (same advice as above)
4. Update the unit test(s) to make sure they still pass muster. Depending on the complexity of the feature, there may be more than one test to update.

Testing
"""""""

Testing is hugely important, as you don’t want your “upgrades” to break the whole package by introducing errors. Thankfully there is a proverbial app for that: *unit testing*. Write a test of your code using the naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case convention
2. function: test_{method}_t{test_id}

(see e.g. test_ui_Series.py for example)

Your test should be as minimal as possible; it is aimed to see if the function your wrote/updated works as advertised given a reasonably comprehensive list of possible arguments. Pyleoclim’s tests rely on data already included in the example_data directory, and we strongly recommend that you do the same; only introduce a new dataset if the existing ones are insufficient to properly test your code. In general, the simpler the test, the better, as it will run in less time and won’t get the Travis gods angry with us.

To run the test(s):
0. Make sure the [pytest package](https://docs.pytest.org) is installed on your system; run `pip install pytest` if not.
1. In your terminal, switch to the “tests” subdirectory of your Pyleoclim forked repository. If you wish to  test a specific class/method inside a specified file, run
`pytest {file_path}::{TestClass}::{test_method}`
2.  To run *all* tests in the specified file, run `pytest {file_path}`
3.  To perform all tests in all testing files inside the specified directory, execute `pytest {directory_path}`

The order above is somewhat loose, but goes from least complex (time*consuming) to more complex.


Stylistic considerations
"""""""""""""""""""""""""
Guido van Rossum’s great insight is that code is read far more often than it is written, so it is important for the code to be of a somewhat uniform style, so that people can read and understand it with relative ease. Pyleoclim strives to use fairly consistent notation, including:

  * capital letters for matrices, lowercase for vectors
  * Independent variable is called ys, the dependent variable  (the time axis) ts.
  * Function names use CamelCase convention

Contributing your changes to Pyleoclim
======================================

Committing your code
""""""""""""""""""""
Once you’ve made changes, you can see them by typing:
:command: `git status`

If you created a new file, it is not being tracked by git. Add it by typing:
:command: `git add path/to/file*to*be*added.py`

Typing :command: `git status` again should give something like:

On branch shiny*new*feature
modified:   /relative/path/to/file*you*added.py

Finally, commit your changes to your local repository with an explanatory message. The message need not be encyclopedic, but it should say what you did, what GitHub issue it refers to, and what part of the code it is expected to affect.
The  preferred style is:

  * a subject line with < 80 chars.
  * One blank line.
  * Optionally, a commit message body.

Now you can commit your changes in your local repository:

:command: `git commit *m 'type your message here'`

Pushing your changes
""""""""""""""""""""

When you want your changes to appear publicly on your GitHub page, push your forked feature branch’s commits:

:command: `git push origin shiny*new*feature`

Here origin is the default name given to your remote repository on GitHub. You can see the remote repositories:

:command: `git remote *v`

If you added the upstream repository as described above you will see something like:

.. code*block:: bash

    origin  git@github.com:yourname/Pyleoclim_util.git (fetch)
    origin  git@github.com:yourname/Pyleoclim_util.git (push)
    upstream  git://github.com/LinkedEarth/Pyleoclim_util.git (fetch)
    upstream  git://github.comLinkedEarth/Pyleoclim_util.git (push)

Now your code is on GitHub, but it is not yet a part of the Pyleoclim project. For that to happen, a pull request needs to be submitted on GitHub.

Filing a Pull Request
"""""""""""""""""""""
When you’re ready to ask for a code review, file a pull request. But before you do, please double*check that you have followed all the guidelines outlined in this document regarding code style, tests, performance tests, and documentation. You should also double check your branch changes against the branch it was based on:

  * Navigate to your repository on GitHub
  * Click on Branches
  * Click on the Compare button for your feature branch
  * Select the base and compare branches, if necessary. This will be *Development* and *shiny*new*feature*, respectively.

If everything looks good, you are ready to make a pull request. A pull request is how code from a local repository becomes available to the GitHub community and can be reviewed by a project’s owners/developers and eventually merged into the master version. This pull request and its associated changes will eventually be committed to the master branch and available in the next release. To submit a pull request:

  * Navigate to your repository on GitHub
  * Click on the Pull Request button
  * You can then click on Commits and Files Changed to make sure everything looks okay one last time
  * Write a description of your changes in the Preview Discussion tab
  * Click Send Pull Request.

This request then goes to the repository maintainers, and they will review the code.

Updating your pull request
""""""""""""""""""""""""""

Based on the review you get on your pull request, you will probably need to make some changes to the code. In that case, you can make them in your branch, add a new commit to that branch, push it to GitHub, and the pull request will be automatically updated. Pushing them to GitHub again is done by:
git push origin shiny*new*feature
This will automatically update your pull request with the latest code and restart the Continuous Integration tests (which is why it is important to provide a test for your code).
Another reason you might need to update your pull request is to solve conflicts with changes that have been merged into the master branch since you opened your pull request.
To do this, you need to “merge upstream master” in your branch:

.. code*block:: bash

    git checkout shiny*new*feature
    git fetch upstream
    git merge upstream/master

If there are no conflicts (or they could be fixed automatically), a file with a default commit message will open, and you can simply save and quit this file.
If there are merge conflicts, you need to solve those conflicts. See [this example](https://help.github.com/articles/resolving*a*merge*conflict*using*the*command*line/) for an explanation on how to do this. Once the conflicts are merged and the files where the conflicts were solved are added, you can run git commit to save those fixes.
If you have uncommitted changes at the moment you want to update the branch with master, you will need to stash them prior to updating (see the stash docs). This will effectively store your changes and they can be reapplied after updating.
After the feature branch has been updated locally, you can now update your pull request by pushing to the branch on GitHub:
git push origin shiny*new*feature

Delete your merged branch (optional)
""""""""""""""""""""""""""""""""""""

Once your feature branch is accepted into upstream, you’ll probably want to get rid of the branch. First, merge upstream master into your branch so git knows it is safe to delete your branch:

.. code*block:: bash

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can do:

:command: `git branch *d shiny*new*feature`

Make sure you use a lower*case *d, or else git won’t warn you if your feature branch has not actually been merged.
The branch will still exist on GitHub, so to delete it there do:

:command: `git push origin **delete shiny*new*feature`

Tips for a successful pull request
""""""""""""""""""""""""""""""""""
If you have made it to the “Review your code” phase, one of the core contributors will take a look. Please note however that response time will be variable (e.g. don’t try the week before AGU).
To improve the chances of your pull request being reviewed, you should:

  * Reference an open issue for non*trivial changes to clarify the PR’s purpose
  * Ensure you have appropriate tests. These should be the first part of any PR
  * Keep your pull requests as simple as possible. Larger PRs take longer to review
  * If you need to add on to what you submitted, keep updating your original pull request, either by request or every few days

Documentation
=============

About the Pyleoclim documentation
"""""""""""""""""""""""""""""""""
Pyleoclim's documentation is built automatically from the function and class docstrings, via [Read The Docs](https://readthedocs.org). It is therefore especially important for your code to include a docstring, and to modify the docstrings of the functions/classes you modified to make sure the documentation is current.

Updating a Pyleoclim docstring
""""""""""""""""""""""""""""""
You may use existing docstrings as examples. A good docstring explains:

  * what the function/class is about
  * what it does, with what properties/inputs/outputs)
  * how to use it, via a minimal working example.

For the latter, make sure the example is:

prefaced by

.. ipython:: python
      :okwarning:
      :okexcept:

    print('done')

and properly indented.

How to build the Pyleoclim documentation
""""""""""""""""""""""""""""""""""""""""

Navigate to the doc_build folder and type :command: `make html`. This may require installing other packages (sphinx, nbsphinx, etc).
