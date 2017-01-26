#!/bin/bash
# Adapted from scikit-learn
# This script is meant to be called in the "deploy" step defined in 
# circle.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the circle.yml in the top level folder of the project.

set -x
set -e


if [ -z $CIRCLE_PROJECT_USERNAME ];
then USERNAME="FreeDiscovery";
else USERNAME=$CIRCLE_PROJECT_USERNAME;
fi

DOC_REPO="freediscovery.github.io"

if [ "$CIRCLE_BRANCH" = "master" ]
then
	dir=dev
else
	# Strip off .X
	dir="${CIRCLE_BRANCH::-2}"
fi

MSG="Pushing the docs to $dir/ for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

cd $HOME
if [ ! -d $DOC_REPO ];
then git clone "git@github.com:FreeDiscovery/"$DOC_REPO".git";
fi
cd $DOC_REPO
git checkout master #$CIRCLE_BRANCH
#git reset --hard origin/$CIRCLE_BRANCH
git rm -rf doc/$dir/ && rm -rf doc/$dir/
mkdir -p doc
cp -R $HOME/FreeDiscovery/doc/_build/html doc/$dir
git config --global user.email "fd@deployment.io"
git config --global user.name "FreeDiscovery"
git config --global push.default matching
git add -f doc/$dir/
git commit -m "$MSG" doc/$dir
git push

echo $MSG 
