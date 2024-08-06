import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib as mpl
import pandas as pd
import sys

def initializePlottingParmaters(fontsize=12):
    mpl.rcParams.update({'font.size': fontsize})
    plt.style.use('seaborn-v0_8-colorblind')
    params = {"ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
            "text.usetex" : True,
            "font.family" : "serif",
            "font.serif" : ["Computer Modern Serif"]}
    plt.rcParams.update(params)

def process_multipleDataPoints(x1, y1, min=True):
	print(x1, y1)
	# kind of a hacky way to do this, but it is ok  
    # Basically if we have two points with the same x value, we want to keep the one with the lowest y value
	data = {'x': x1, 'y': y1}
	df = pd.DataFrame(data)
	if min:
		df = df.groupby(['x'], as_index=False).min()
	else:
		df = df.groupby(['x'], as_index=False).max()
	print(df['x'].values, df['y'].values)
	return df['x'].values, df['y'].values

def process_inv(r):
	# This function replaces any inverse designs with the previous value if it is bigger 
    # Useful when plotting inverse design perf as a function of a parameter that strictly increases the objective as it increases
	for i in range(1, len(r)):
		if (max and (r[i] < r[i-1]) and (r[i-1] < np.inf)):
			r[i] = r[i-1]
	return r

# def replace_larger_than_last(x1, y1):
# 	# This function replaces any values larger than the last value with the last value
# 	y1[y1 > y1[-1]] = y1[-1]
# 	return x1, y1

def get_highest_gpr_xs_ys(data):
	xs = np.sort(data['dx'].unique())
	ys = np.zeros(len(xs))
	gprs = np.zeros(len(xs))
	nprojs = np.zeros(len(xs))
	for idx, dx in enumerate(xs):
		temp = data[data['dx'] == dx]
		temp = temp[temp['gpr'] == np.max(temp['gpr'].values)]
		ys[idx] = np.min(temp['bound'].values)
		gprs[idx] = np.max(temp['gpr'].values)
		nprojs[idx] = temp[(temp['bound'] == ys[idx]) & (temp['gpr'] == gprs[idx])]['nprojx'].values[0]

	return xs, process_inv(ys), gprs, nprojs

def shapeInverseDesignPlots(axis, set_square=True):
	axis.get_yaxis().set_visible(False)
	axis.get_xaxis().set_visible(False)
	if set_square: axis.set_box_aspect(1)

def sort_by_x(x, y):
	print(x, y)
	indx = x.argsort()
	return x[indx], y[indx]