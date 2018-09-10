#import python dependencies
import os, shutil, sys, socket, platform

#import third-party dependencies
import requests
from PyQt4 import QtGui, QtCore

#settings
caption = 'FreeDiscovery Installer'
width = 520
height = 240

deps = {	'python':'Python',
			'numpy':'Numpy',
			'scipy':'Scipy',
			'scikit':'SciKit Learn' }

#system
app = None
window = None
font = None


#network
def dl_loc (name):
	get = os.getcwd().replace('\\', '/')
	if '/' not in get: get = '.'
	return get + '/' + name

def download (From, to):
	dfile = open(dl_loc(to), 'wb')
	req = requests.get(From, stream=True)
	raw = req.raw
	while True:
		get = raw.read()
		if get: dfile.write(get)
		else: break

#gui
widgets = []

def label (x, y, text):
	new = QtGui.QLabel(window)
	new.setText(text)
	new.setFont(font)
	new.move(x, y)
	new.show()
	widgets.append(new)
	return new

section_x = 10
section_y = 10
def section (text, x = 0):
	global section_x, section_y
	sect = label(section_x + x, section_y, text)
	section_y += sect.height()
	return sect

class check_section:
	def __init__(c, checked, text):
		c.checked = checked
		c.text = text
		c.sect = section(c.text)
	def yes (c): c.checked = True
	def no (c): c.checked = False

def bar (range_max=10):
	new = QtGui.QProgressBar(window)
	new.resize(width - 40, 20)
	new.move(20, height - 50)
	new.setValue(0)
	new.setRange(0, range_max)
	new.show()
	widgets.append(new)
	return new

def logo ():
	pic = QtGui.QLabel(window)
	#pic.setGeometry(10, 10, 400, 200)
	pixmap = QtGui.QPixmap(logo_pix)
	#pixmap = pixmap.scaledToHeight(200)
	pic.setPixmap(pixmap)
	pic.show()
	widgets.append(pic)
	return pic

def description (text):
	new = label(0, 0, ''.join([line.strip() + '\n' for line in text.strip().split('\n')]).replace('TAB', '\t'))
	return new
	
def next_button (text = 'Next'):
	global width, height
	new = QtGui.QPushButton(window)
	new.setText(text)
	new.move(width - new.width(), height - new.height())
	new.show()
	widgets.append(new)
	return new


#main program structure
class error(Exception):
	def __init__(e, message): e.message = message
	def __str__(e): return e.message

glos = {}
state_keys = []
state_funcs = {}
def state (func):
	
	'''
	Decorator function to be able to expand later if needed. Right now,
	functions are added. Each function represents a different "state" of
	the installer. There is a "start" function for win the state starts,
	and a "run" function for when the function is running.
	'''
	
	name = func.__name__.replace('_start', '').replace('_run', '')
	if name not in state_keys: state_keys.append(name)
	
	if name not in state_funcs: state_funcs[name] = {'mode':'start', 'values':{}}
	state_funcs[name][func.__name__.replace(name + '_', '')] = func

	return func

def clear ():
	global sections, section_y
	state_keys.pop(0)
	while widgets:
		widget = widgets.pop(0)
		widget.setParent(None)
		widget.destroy()
	sections = []
	section_y = 10

def main ():
	global widgets, section_y, state_keys
	
	'''
	This is the main function for the installer. It runs all
	of the state funcs saved. Designed to be modular.
	'''
	
	#check if it's still going
	if state_keys:
		
		#run
		try:
			dic = state_funcs[state_keys[0]]
			if dic[dic['mode']]( dic['values'] ):
				if dic['mode'] == 'start': dic['mode'] = 'run'
				else: clear()
		
		#error occurred!
		except error as e:
			clear()
			state_keys = ['error']
			state_funcs['error'] = {
				'values': {'error' : e.message},
				'mode': 'start',
				'start': error_start,
				'run': error_run,
			}
	
	#nope!
	else: app.exit(1)


#error screen
def error_start (values):
	section('Error:')
	section(values['error'])
	section('\nPlease try the installer again.')
	values['exit'] = next_button('Exit')
	return True
	
def error_run (values): return values['exit'].isDown()


#splash screen
@state
def splash_start (values):
	
	#has admin rights?
	if 'lin' in sys.platform and os.getuid() > 0:
		raise error('This installer needs admin rights to run (sudo)')
	
	#page
	'The opening of the installer, introduces the project'
	section('FreeDiscovery installer...')
	values['next'] = next_button()
	
	#finish
	return True

@state
def splash_run (values): return values['next'].isDown()


#check dependencies splash
@state
def check_deps_splash_start (values):
	section('Going to check for any dependencies that you may')
	section('already have.')
	section('* Python', 10)
	section('* Numpy', 10)
	section('* SciPy', 10)
	section('* SciKit Learn', 10)
	section('Press "Next" to start the scan.', 10)
	values['next'] = next_button()
	return True

@state
def check_deps_splash_run (values): return values['next'].isDown()


#checking dependencies
@state
def check_deps_start (values):
	section('Checking for dependencies')
	
	for dep in deps:
		values[dep] = check_section(False, deps[dep])
		glos[dep + '_checked'] = False
		glos[dep] = False

	values['bar'] = bar(len(deps))
	values['mode'] = ''
	
	return True
	
@state
def check_deps_run (values):
	
	#start
	if values['mode'] == '':
		values['mode'] = 'python'
		return False

	#check for python
	elif values['mode'] == 'python':
		
		#has python
		if glos['python'] and os.path.exists('C:/Python27'):
			values['bar'].setValue(1)
			glos['python_check'] = True
			values['mode'] = 'numpy'
			return False
		
		#else
		else:
			for dep in deps: values[dep + '_check'] = True
			return True

	#check for numpy
	elif values['mode'] == 'numpy':
		values['bar'].setValue(2)
		glos['numpy_check'] = True
		values['mode'] = 'scipy'
		return False

	#check for scipy
	elif values['mode'] == 'scipy':
		values['bar'].setValue(3)
		glos['scipy_check'] = True
		values['mode'] = 'scikit'
		return False

	#check for scikit
	elif values['mode'] == 'scikit':
		values['bar'].setValue(4)
		glos['scikit_check'] = True
		return True


#
def _install_dep (name):
	if sys.platform.startswith('win'): os.system('C:\\Python35\\Scripts\\pip.exe install ' + name)
	else: os.system('pip install ' + name)

@state
def install_deps_start (values):

	#needs some dependencies
	if any([not glos[name] for name in deps]):
		glos['skip'] = False
		section('Will now install the following dependencies:')
		for name in deps:
			if not glos[name]: section(deps[name])
			
	#has everything!
	else:
		glos['skip'] = True
		section('Looks like all dependencies are installed!')
		section('Click "Next" to install FreeDiscovery.')
	
	#finished
	values['mode'] = ''
	values['bar'] = bar(len(deps))
	values['next'] = next_button()
	return True

@state
def install_deps_run (values):
	
	#
	if glos['skip']: return True
	else:
		
		#start
		if values['mode'] == '':			
			values['mode'] = 'python'
			return False

		#check for python
		elif values['mode'] == 'python':
			
			#install linux
			if 'linux' in sys.platform: os.system('apt-get install python')
			
			#install for windows
			else:
				if 'darwin' in sys.platform:
					url = '3.5.0/python-3.5.0-macosx10.6.pkg'
					name = 'pyinstall.pkg'
				elif 'win' in sys.platform:
					if '64' in platform.architecture()[0]: url = '3.5.0/python-3.5.0-amd64.exe'
					else: url = '3.5.0/python-3.5.0.exe'
					name = 'pyinstall.exe'
					os.environ['path'] = os.environ['path'] + ';C:\\Python35'
				else: raise error('Platform not supported')
				py = download('https://www.python.org/ftp/python/' + url, name)
				os.system(dl_loc(name))
				os.remove(dl_loc(name))

			#progress
			values['bar'].setValue(1)
			glos['python_check'] = True
			values['mode'] = 'numpy'
			return False

		#check for numpy
		elif values['mode'] == 'numpy':

			_install_dep ('numpy')

			#
			values['bar'].setValue(2)
			glos['numpy_check'] = True
			values['mode'] = 'scipy'
			return False

		#check for scipy
		elif values['mode'] == 'scipy':
			_install_dep('scipy')
			values['bar'].setValue(3)
			glos['scipy_check'] = True
			values['mode'] = 'scikit'
			return False

		#check for scikit
		elif values['mode'] == 'scikit':
			_install_dep('scikit-learn')
			values['bar'].setValue(4)
			glos['scikit_check'] = True
			return True
#
@state
def install_freedisc_start (values):
	section('Now installing FreeDiscovery...')
	return True

@state
def install_freedisc_run (values):
	_install_dep('freediscovery')
	return True


#
@state
def complete_start (values):
	section('FreeDiscovery has now been installed!')
	section('Press "Exit" to finish')
	values['exit'] = next_button('Exit')
	return True

@state
def complete_run (values): return values['exit'].isDown()


#main program
if __name__ == '__main__':
	
	#create QT app
	app = QtGui.QApplication([])
	
	#create Window
	window = QtGui.QWidget()
	window.resize(width, height)
	window.setWindowTitle(caption)
	window.show()

	#make font
	font = QtGui.QFont()
	font.setPointSize(14)

	#start process
	process = QtCore.QTimer()
	process.timeout.connect(main)
	process.start(1)
	
	#execute!
	app.exec_()
