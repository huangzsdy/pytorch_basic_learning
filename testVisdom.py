from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
from sys import platform as _platform
from six.moves import urllib

viz = Visdom(port=8098,env='main')

assert viz.check_connection()
viz.close()


win = viz.line(
	X = np.array([0,1]),
	Y = np.array([0,1]),
	opts = dict(
		# xtickmin = -2,
		# xtickmax = 2,
		# xtickstep = 1,
		# ytickmin = -3,
		# ytickmax = 5,
		# ytickstep = 1,
		markersysmbol = 'dot',
		markersize = 5,
		showlegend = False,
		),
	name = '1'
)

viz.line(
	X = np.array([0,1]),
	Y = np.array([1,2]),
	opts = dict(markercolor = np.array([50]),markersysmbol = 'dot',),
	win = win,
	update = 'new',
	name = '2',
	)
for i in range(10000):
	viz.line(
		X = np.array([i]),
		Y = np.array([i * 2]),
		win = win,
		name = '1',
		update='append'
		)
	viz.line(
		X = np.array([i]),
		Y = np.array([i*10]),
		win = win,
		name = '2',
		update='append'
		)	
