#!/bin/bash

# Stop upon any errors encountered.
set -o errexit

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd $SCRIPT_DIR

# Default args
keep=0
force=0
dry=0

usage() {
    cat <<END
Usage: `basename $0` [OPTIONS]

    Update the "docs" branch of this repo with newly generated
    documentation from the current source. A new commit is created in the
    current (local) repo, which can afterwards be pushed online using:

        git push origin docs

OPTIONS:
    -f/--force  Remove any previoud temporary "docs_repo" directory.
    -k/--keep   Keep the temporary "docs_repo" directory.
    -d/--dry    Do not push to docs branch of current repo.
    -h/--help   Show this help message
END
}

# NOTE: To create the initial (empty) docs branch, we used:
#
#   $ git checkout --orphan docs
#   $ git rm -rf .
#   $ touch .nojekyll
#   $ git add .nojekyll
#   $ ... edit and add e.g. .gitignore ...
#   $ git commit -m "Create empty docs branch"
#   $ git push -u origin docs
#
# This only needs to be done *once* per project.


while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -d|--dry) dry=1 ;;
        -k|--keep) keep=1 ;;
        -f|--force) force=1 ;;
        -h|--help) usage; exit ;;
        *) echo "ERROR: Unknown argument: $1"; usage; exit 1 ;;
    esac
    shift
done

git_branch=$( git rev-parse --abbrev-ref HEAD )
if [[ "$git_branch" != "master" ]]; then
    echo "ERROR: Not on master branch."
    exit 1
fi

git_revision=$( git describe --tags --always --dirty )
echo "Generating documentation for revision $git_revision"

if [ -d "docs_repo" ]; then
    if [ $force -eq 1 ]; then
        rm -rf docs_repo
    else
        echo "ERROR: 'docs_repo' directory already exists." >&2
        exit 1
    fi
fi

# Create temporary folder (will be deleted later)
mkdir docs_repo
cd docs_repo

# Create a temporary Doxyfile based on main Doxyfile
cat <<EOF > Doxyfile_docs
@INCLUDE = Doxyfile
OUTPUT_DIRECTORY       = ./docs_repo/repo/docs
GENERATE_HTML          = YES
HTML_OUTPUT            = .
GENERATE_LATEX         = NO
PROJECT_NUMBER         = $git_revision
EOF

# Clone the local repo and checkout our docs branch
git clone -b docs .. repo
cd repo

# Clean previous docs
rm -rf docs
mkdir docs
touch docs/.nojekyll

# (Re-)create the actual documentation
cd ../../
doxygen docs_repo/Doxyfile_docs 2>&1 | tee docs_repo/doxygen.log
cd docs_repo/repo

# Clean and udpate auxiliary files
rm -rf docs_input
mkdir docs_input
cp ../../docs_input/*.ipynb docs_input/
cp ../../docs_input/*.html docs_input/

# Commit, push to local repo and delete temporary clone
if [ -d "docs" -a -f "docs/index.html" ]; then
    echo "Documentation created. Committing..."
    git add --all
    git commit -m "Update docs for $git_revision"
    if [ $dry -eq 0 ]; then
        git push > /dev/null 2>&1
        echo "Changes pushed to local repo"
    fi
    cd ../../
    if [ $keep -eq 0 ]; then
        echo "Removing temporary files"
        rm -rf docs_repo
    fi
else
    echo "ERROR: Doxygen did not create documentation." >&2
    exit 1
fi
echo "Done"
if [ $dry -eq 0 ]; then
    echo
    echo "You may now push the new commit using:"
    echo "    git push origin docs"
fi
