sdist:
	make stamp; python setup.py sdist --formats=zip; 

stamp:
	sed -i '$$ d' freediscovery/__init__.py ; sed -i '$$ d' freediscovery/__init__.py ; echo '__version_date__ = "'`git log --pretty=format:'%cd' -n 1`'"' >> freediscovery/__init__.py; echo '__version_hash__ = "'`git log --pretty=format:'%h' -n 1`'"' >> freediscovery/__init__.py

test2: 
	cd ..; python2 -c "import freediscovery.tests as ft; ft.run()"
test3: 
	cd ..; python3 -c "import freediscovery.tests as ft; ft.run()"

test: test3

test_cov: 
	cd ..; python3 -c "import freediscovery.tests as ft; ft.run(coverage=True)"
