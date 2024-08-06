#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 20:23:03 2022

@author: pengning
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import quad as np_quad
from scipy.optimize import minimize_scalar

import matplotlib.pyplot as plt

import sympy as sym
from sympy.parsing.sympy_parser import parse_expr
import mpmath
from mpmath import mp

mp.dps = 25

str_denom = '4*(kyi - I*kyr)*(kyi + I*kyr)*(mf + pi)*(kyi^4*mf + 2*kyi^2*kyr^2*mf + kyr^4*mf - kx^2*kyi^2*pi + kyi^4*pi + kx^2*kyr^2*pi + 2*kyi^2*kyr^2*pi + kyr^4*pi + 2*kx^2*kyi*kyr*pr) + 4*(kyi - I*kyr)*(kyi + I*kyr)*(mf + pi)*(-2*kyi^2*mf + 2*kyr^2*mf + kx^2*pi - kyi^2*pi + kyr^2*pi - 2*kyi*kyr*pr)*s^2 + 4*(kyi - I*kyr)*(kyi + I*kyr)*mf*(mf + pi)*s^4'
str_denom = str_denom.replace('^', '**')


str_Fxs_num = '-4*kx*kyi^5*mf - (4*I)*kx*kyi^4*kyr*mf - 8*kx*kyi^3*kyr^2*mf - (8*I)*kx*kyi^2*kyr^3*mf - 4*kx*kyi*kyr^4*mf - (4*I)*kx*kyr^5*mf + 4*kx^3*kyi^3*pi - 4*kx*kyi^5*pi + (4*I)*kx^3*kyi^2*kyr*pi - (4*I)*kx*kyi^4*kyr*pi - 4*kx^3*kyi*kyr^2*pi - 8*kx*kyi^3*kyr^2*pi - (4*I)*kx^3*kyr^3*pi - (8*I)*kx*kyi^2*kyr^3*pi - 4*kx*kyi*kyr^4*pi - (4*I)*kx*kyr^5*pi - 8*kx^3*kyi^2*kyr*pr - (8*I)*kx^3*kyi*kyr^2*pr + 4*kx*kyi^4*mf*s + (8*I)*kx*kyi^3*kyr*mf*s + (8*I)*kx*kyi*kyr^3*mf*s - 4*kx*kyr^4*mf*s + 4*kx*kyi^4*pi*s + (8*I)*kx*kyi^3*kyr*pi*s + (8*I)*kx*kyi*kyr^3*pi*s - 4*kx*kyr^4*pi*s + 4*kx*kyi^3*mf*s^2 - (4*I)*kx*kyi^2*kyr*mf*s^2 + 4*kx*kyi*kyr^2*mf*s^2 - (4*I)*kx*kyr^3*mf*s^2 - 4*kx^3*kyi*pi*s^2 + 4*kx*kyi^3*pi*s^2 - (4*I)*kx^3*kyr*pi*s^2 - (4*I)*kx*kyi^2*kyr*pi*s^2 + 4*kx*kyi*kyr^2*pi*s^2 - (4*I)*kx*kyr^3*pi*s^2 - 4*kx*kyi^2*mf*s^3 - 4*kx*kyr^2*mf*s^3 - 4*kx*kyi^2*pi*s^3 - 4*kx*kyr^2*pi*s^3 + Fym*((-I)*kx*kyi^5*mf*pi + kx*kyi^4*kyr*mf*pi - (2*I)*kx*kyi^3*kyr^2*mf*pi + 2*kx*kyi^2*kyr^3*mf*pi - I*kx*kyi*kyr^4*mf*pi + kx*kyr^5*mf*pi + I*kx^3*kyi^3*pi^2 - I*kx*kyi^5*pi^2 - kx^3*kyi^2*kyr*pi^2 + kx*kyi^4*kyr*pi^2 - I*kx^3*kyi*kyr^2*pi^2 - (2*I)*kx*kyi^3*kyr^2*pi^2 + kx^3*kyr^3*pi^2 + 2*kx*kyi^2*kyr^3*pi^2 - I*kx*kyi*kyr^4*pi^2 + kx*kyr^5*pi^2 + kx*kyi^5*mf*pr + I*kx*kyi^4*kyr*mf*pr + 2*kx*kyi^3*kyr^2*mf*pr + (2*I)*kx*kyi^2*kyr^3*mf*pr + kx*kyi*kyr^4*mf*pr + I*kx*kyr^5*mf*pr - kx^3*kyi^3*pi*pr + kx*kyi^5*pi*pr - (3*I)*kx^3*kyi^2*kyr*pi*pr + I*kx*kyi^4*kyr*pi*pr + 3*kx^3*kyi*kyr^2*pi*pr + 2*kx*kyi^3*kyr^2*pi*pr + I*kx^3*kyr^3*pi*pr + (2*I)*kx*kyi^2*kyr^3*pi*pr + kx*kyi*kyr^4*pi*pr + I*kx*kyr^5*pi*pr + 2*kx^3*kyi^2*kyr*pr^2 + (2*I)*kx^3*kyi*kyr^2*pr^2 - I*kx*kyi^4*mf*pi*s + 2*kx*kyi^3*kyr*mf*pi*s + 2*kx*kyi*kyr^3*mf*pi*s + I*kx*kyr^4*mf*pi*s - I*kx*kyi^4*pi^2*s + 2*kx*kyi^3*kyr*pi^2*s + 2*kx*kyi*kyr^3*pi^2*s + I*kx*kyr^4*pi^2*s + kx*kyi^4*mf*pr*s + (2*I)*kx*kyi^3*kyr*mf*pr*s + (2*I)*kx*kyi*kyr^3*mf*pr*s - kx*kyr^4*mf*pr*s + kx*kyi^4*pi*pr*s + (2*I)*kx*kyi^3*kyr*pi*pr*s + (2*I)*kx*kyi*kyr^3*pi*pr*s - kx*kyr^4*pi*pr*s + I*kx*kyi^3*mf*pi*s^2 + kx*kyi^2*kyr*mf*pi*s^2 + I*kx*kyi*kyr^2*mf*pi*s^2 + kx*kyr^3*mf*pi*s^2 - I*kx^3*kyi*pi^2*s^2 + I*kx*kyi^3*pi^2*s^2 + kx^3*kyr*pi^2*s^2 + kx*kyi^2*kyr*pi^2*s^2 + I*kx*kyi*kyr^2*pi^2*s^2 + kx*kyr^3*pi^2*s^2 - kx*kyi^3*mf*pr*s^2 + I*kx*kyi^2*kyr*mf*pr*s^2 - kx*kyi*kyr^2*mf*pr*s^2 + I*kx*kyr^3*mf*pr*s^2 + kx^3*kyi*pi*pr*s^2 - kx*kyi^3*pi*pr*s^2 + I*kx^3*kyr*pi*pr*s^2 + I*kx*kyi^2*kyr*pi*pr*s^2 - kx*kyi*kyr^2*pi*pr*s^2 + I*kx*kyr^3*pi*pr*s^2 + I*kx*kyi^2*mf*pi*s^3 + I*kx*kyr^2*mf*pi*s^3 + I*kx*kyi^2*pi^2*s^3 + I*kx*kyr^2*pi^2*s^3 - kx*kyi^2*mf*pr*s^3 - kx*kyr^2*mf*pr*s^3 - kx*kyi^2*pi*pr*s^3 - kx*kyr^2*pi*pr*s^3) + Fyc*((-I)*kx*kyi^5*mf*pi - kx*kyi^4*kyr*mf*pi - (2*I)*kx*kyi^3*kyr^2*mf*pi - 2*kx*kyi^2*kyr^3*mf*pi - I*kx*kyi*kyr^4*mf*pi - kx*kyr^5*mf*pi + I*kx^3*kyi^3*pi^2 - I*kx*kyi^5*pi^2 + kx^3*kyi^2*kyr*pi^2 - kx*kyi^4*kyr*pi^2 - I*kx^3*kyi*kyr^2*pi^2 - (2*I)*kx*kyi^3*kyr^2*pi^2 - kx^3*kyr^3*pi^2 - 2*kx*kyi^2*kyr^3*pi^2 - I*kx*kyi*kyr^4*pi^2 - kx*kyr^5*pi^2 - kx*kyi^5*mf*pr + I*kx*kyi^4*kyr*mf*pr - 2*kx*kyi^3*kyr^2*mf*pr + (2*I)*kx*kyi^2*kyr^3*mf*pr - kx*kyi*kyr^4*mf*pr + I*kx*kyr^5*mf*pr + kx^3*kyi^3*pi*pr - kx*kyi^5*pi*pr - (3*I)*kx^3*kyi^2*kyr*pi*pr + I*kx*kyi^4*kyr*pi*pr - 3*kx^3*kyi*kyr^2*pi*pr - 2*kx*kyi^3*kyr^2*pi*pr + I*kx^3*kyr^3*pi*pr + (2*I)*kx*kyi^2*kyr^3*pi*pr - kx*kyi*kyr^4*pi*pr + I*kx*kyr^5*pi*pr - 2*kx^3*kyi^2*kyr*pr^2 + (2*I)*kx^3*kyi*kyr^2*pr^2 - I*kx*kyi^4*mf*pi*s - 2*kx*kyi^3*kyr*mf*pi*s - 2*kx*kyi*kyr^3*mf*pi*s + I*kx*kyr^4*mf*pi*s - I*kx*kyi^4*pi^2*s - 2*kx*kyi^3*kyr*pi^2*s - 2*kx*kyi*kyr^3*pi^2*s + I*kx*kyr^4*pi^2*s - kx*kyi^4*mf*pr*s + (2*I)*kx*kyi^3*kyr*mf*pr*s + (2*I)*kx*kyi*kyr^3*mf*pr*s + kx*kyr^4*mf*pr*s - kx*kyi^4*pi*pr*s + (2*I)*kx*kyi^3*kyr*pi*pr*s + (2*I)*kx*kyi*kyr^3*pi*pr*s + kx*kyr^4*pi*pr*s + I*kx*kyi^3*mf*pi*s^2 - kx*kyi^2*kyr*mf*pi*s^2 + I*kx*kyi*kyr^2*mf*pi*s^2 - kx*kyr^3*mf*pi*s^2 - I*kx^3*kyi*pi^2*s^2 + I*kx*kyi^3*pi^2*s^2 - kx^3*kyr*pi^2*s^2 - kx*kyi^2*kyr*pi^2*s^2 + I*kx*kyi*kyr^2*pi^2*s^2 - kx*kyr^3*pi^2*s^2 + kx*kyi^3*mf*pr*s^2 + I*kx*kyi^2*kyr*mf*pr*s^2 + kx*kyi*kyr^2*mf*pr*s^2 + I*kx*kyr^3*mf*pr*s^2 - kx^3*kyi*pi*pr*s^2 + kx*kyi^3*pi*pr*s^2 + I*kx^3*kyr*pi*pr*s^2 + I*kx*kyi^2*kyr*pi*pr*s^2 + kx*kyi*kyr^2*pi*pr*s^2 + I*kx*kyr^3*pi*pr*s^2 + I*kx*kyi^2*mf*pi*s^3 + I*kx*kyr^2*mf*pi*s^3 + I*kx*kyi^2*pi^2*s^3 + I*kx*kyr^2*pi^2*s^3 + kx*kyi^2*mf*pr*s^3 + kx*kyr^2*mf*pr*s^3 + kx*kyi^2*pi*pr*s^3 + kx*kyr^2*pi*pr*s^3) + Fxm*(kyi^6*mf*pi + 3*kyi^4*kyr^2*mf*pi + 3*kyi^2*kyr^4*mf*pi + kyr^6*mf*pi - kx^2*kyi^4*pi^2 + kyi^6*pi^2 + 3*kyi^4*kyr^2*pi^2 + kx^2*kyr^4*pi^2 + 3*kyi^2*kyr^4*pi^2 + kyr^6*pi^2 + I*kyi^6*mf*pr + (3*I)*kyi^4*kyr^2*mf*pr + (3*I)*kyi^2*kyr^4*mf*pr + I*kyr^6*mf*pr - I*kx^2*kyi^4*pi*pr + I*kyi^6*pi*pr + 2*kx^2*kyi^3*kyr*pi*pr + (3*I)*kyi^4*kyr^2*pi*pr + 2*kx^2*kyi*kyr^3*pi*pr + I*kx^2*kyr^4*pi*pr + (3*I)*kyi^2*kyr^4*pi*pr + I*kyr^6*pi*pr + (2*I)*kx^2*kyi^3*kyr*pr^2 + (2*I)*kx^2*kyi*kyr^3*pr^2 + kyi^5*mf*pi*s + I*kyi^4*kyr*mf*pi*s + 2*kyi^3*kyr^2*mf*pi*s + (2*I)*kyi^2*kyr^3*mf*pi*s + kyi*kyr^4*mf*pi*s + I*kyr^5*mf*pi*s + kyi^5*pi^2*s + I*kyi^4*kyr*pi^2*s + 2*kyi^3*kyr^2*pi^2*s + (2*I)*kyi^2*kyr^3*pi^2*s + kyi*kyr^4*pi^2*s + I*kyr^5*pi^2*s + I*kyi^5*mf*pr*s - kyi^4*kyr*mf*pr*s + (2*I)*kyi^3*kyr^2*mf*pr*s - 2*kyi^2*kyr^3*mf*pr*s + I*kyi*kyr^4*mf*pr*s - kyr^5*mf*pr*s + I*kyi^5*pi*pr*s - kyi^4*kyr*pi*pr*s + (2*I)*kyi^3*kyr^2*pi*pr*s - 2*kyi^2*kyr^3*pi*pr*s + I*kyi*kyr^4*pi*pr*s - kyr^5*pi*pr*s - kyi^4*mf*pi*s^2 + (2*I)*kyi^3*kyr*mf*pi*s^2 + (2*I)*kyi*kyr^3*mf*pi*s^2 + kyr^4*mf*pi*s^2 + kx^2*kyi^2*pi^2*s^2 - kyi^4*pi^2*s^2 + (2*I)*kyi^3*kyr*pi^2*s^2 + kx^2*kyr^2*pi^2*s^2 + (2*I)*kyi*kyr^3*pi^2*s^2 + kyr^4*pi^2*s^2 - I*kyi^4*mf*pr*s^2 - 2*kyi^3*kyr*mf*pr*s^2 - 2*kyi*kyr^3*mf*pr*s^2 + I*kyr^4*mf*pr*s^2 + I*kx^2*kyi^2*pi*pr*s^2 - I*kyi^4*pi*pr*s^2 - 2*kyi^3*kyr*pi*pr*s^2 + I*kx^2*kyr^2*pi*pr*s^2 - 2*kyi*kyr^3*pi*pr*s^2 + I*kyr^4*pi*pr*s^2 - kyi^3*mf*pi*s^3 + I*kyi^2*kyr*mf*pi*s^3 - kyi*kyr^2*mf*pi*s^3 + I*kyr^3*mf*pi*s^3 - kyi^3*pi^2*s^3 + I*kyi^2*kyr*pi^2*s^3 - kyi*kyr^2*pi^2*s^3 + I*kyr^3*pi^2*s^3 - I*kyi^3*mf*pr*s^3 - kyi^2*kyr*mf*pr*s^3 - I*kyi*kyr^2*mf*pr*s^3 - kyr^3*mf*pr*s^3 - I*kyi^3*pi*pr*s^3 - kyi^2*kyr*pi*pr*s^3 - I*kyi*kyr^2*pi*pr*s^3 - kyr^3*pi*pr*s^3) + Fxc*(kyi^6*mf*pi + 3*kyi^4*kyr^2*mf*pi + 3*kyi^2*kyr^4*mf*pi + kyr^6*mf*pi - kx^2*kyi^4*pi^2 + kyi^6*pi^2 + 3*kyi^4*kyr^2*pi^2 + kx^2*kyr^4*pi^2 + 3*kyi^2*kyr^4*pi^2 + kyr^6*pi^2 - I*kyi^6*mf*pr - (3*I)*kyi^4*kyr^2*mf*pr - (3*I)*kyi^2*kyr^4*mf*pr - I*kyr^6*mf*pr + I*kx^2*kyi^4*pi*pr - I*kyi^6*pi*pr + 2*kx^2*kyi^3*kyr*pi*pr - (3*I)*kyi^4*kyr^2*pi*pr + 2*kx^2*kyi*kyr^3*pi*pr - I*kx^2*kyr^4*pi*pr - (3*I)*kyi^2*kyr^4*pi*pr - I*kyr^6*pi*pr - (2*I)*kx^2*kyi^3*kyr*pr^2 - (2*I)*kx^2*kyi*kyr^3*pr^2 + kyi^5*mf*pi*s - I*kyi^4*kyr*mf*pi*s + 2*kyi^3*kyr^2*mf*pi*s - (2*I)*kyi^2*kyr^3*mf*pi*s + kyi*kyr^4*mf*pi*s - I*kyr^5*mf*pi*s + kyi^5*pi^2*s - I*kyi^4*kyr*pi^2*s + 2*kyi^3*kyr^2*pi^2*s - (2*I)*kyi^2*kyr^3*pi^2*s + kyi*kyr^4*pi^2*s - I*kyr^5*pi^2*s - I*kyi^5*mf*pr*s - kyi^4*kyr*mf*pr*s - (2*I)*kyi^3*kyr^2*mf*pr*s - 2*kyi^2*kyr^3*mf*pr*s - I*kyi*kyr^4*mf*pr*s - kyr^5*mf*pr*s - I*kyi^5*pi*pr*s - kyi^4*kyr*pi*pr*s - (2*I)*kyi^3*kyr^2*pi*pr*s - 2*kyi^2*kyr^3*pi*pr*s - I*kyi*kyr^4*pi*pr*s - kyr^5*pi*pr*s - kyi^4*mf*pi*s^2 - (2*I)*kyi^3*kyr*mf*pi*s^2 - (2*I)*kyi*kyr^3*mf*pi*s^2 + kyr^4*mf*pi*s^2 + kx^2*kyi^2*pi^2*s^2 - kyi^4*pi^2*s^2 - (2*I)*kyi^3*kyr*pi^2*s^2 + kx^2*kyr^2*pi^2*s^2 - (2*I)*kyi*kyr^3*pi^2*s^2 + kyr^4*pi^2*s^2 + I*kyi^4*mf*pr*s^2 - 2*kyi^3*kyr*mf*pr*s^2 - 2*kyi*kyr^3*mf*pr*s^2 - I*kyr^4*mf*pr*s^2 - I*kx^2*kyi^2*pi*pr*s^2 + I*kyi^4*pi*pr*s^2 - 2*kyi^3*kyr*pi*pr*s^2 - I*kx^2*kyr^2*pi*pr*s^2 - 2*kyi*kyr^3*pi*pr*s^2 - I*kyr^4*pi*pr*s^2 - kyi^3*mf*pi*s^3 - I*kyi^2*kyr*mf*pi*s^3 - kyi*kyr^2*mf*pi*s^3 - I*kyr^3*mf*pi*s^3 - kyi^3*pi^2*s^3 - I*kyi^2*kyr*pi^2*s^3 - kyi*kyr^2*pi^2*s^3 - I*kyr^3*pi^2*s^3 + I*kyi^3*mf*pr*s^3 - kyi^2*kyr*mf*pr*s^3 + I*kyi*kyr^2*mf*pr*s^3 - kyr^3*mf*pr*s^3 + I*kyi^3*pi*pr*s^3 - kyi^2*kyr*pi*pr*s^3 + I*kyi*kyr^2*pi*pr*s^3 - kyr^3*pi*pr*s^3)'
str_Fxs_num = str_Fxs_num.replace('^', '**')

str_Fys_num = '-(kx*((4*I)*kx*kyi^4*mf - 8*kx*kyi^3*kyr*mf - 8*kx*kyi*kyr^3*mf - (4*I)*kx*kyr^4*mf + (4*I)*kx*kyi^4*pi - 8*kx*kyi^3*kyr*pi - 8*kx*kyi*kyr^3*pi - (4*I)*kx*kyr^4*pi - (4*I)*kx*kyi^3*mf*s + 12*kx*kyi^2*kyr*mf*s + (12*I)*kx*kyi*kyr^2*mf*s - 4*kx*kyr^3*mf*s + 8*kx*kyi^2*kyr*pi*s + (8*I)*kx*kyi*kyr^2*pi*s - (8*I)*kx*kyi^2*kyr*pr*s + 8*kx*kyi*kyr^2*pr*s - (4*I)*kx*kyi^2*mf*s^2 - (4*I)*kx*kyr^2*mf*s^2 - (4*I)*kx*kyi^2*pi*s^2 - (4*I)*kx*kyr^2*pi*s^2 + (4*I)*kx*kyi*mf*s^3 - 4*kx*kyr*mf*s^3)) - Fym*kx*(kx*kyi^4*mf*pi + (2*I)*kx*kyi^3*kyr*mf*pi + (2*I)*kx*kyi*kyr^3*mf*pi - kx*kyr^4*mf*pi + kx*kyi^4*pi^2 + (2*I)*kx*kyi^3*kyr*pi^2 + (2*I)*kx*kyi*kyr^3*pi^2 - kx*kyr^4*pi^2 + I*kx*kyi^4*mf*pr - 2*kx*kyi^3*kyr*mf*pr - 2*kx*kyi*kyr^3*mf*pr - I*kx*kyr^4*mf*pr + I*kx*kyi^4*pi*pr - 2*kx*kyi^3*kyr*pi*pr - 2*kx*kyi*kyr^3*pi*pr - I*kx*kyr^4*pi*pr + kx*kyi^3*mf*pi*s + (3*I)*kx*kyi^2*kyr*mf*pi*s - 3*kx*kyi*kyr^2*mf*pi*s - I*kx*kyr^3*mf*pi*s + (2*I)*kx*kyi^2*kyr*pi^2*s - 2*kx*kyi*kyr^2*pi^2*s + I*kx*kyi^3*mf*pr*s - 3*kx*kyi^2*kyr*mf*pr*s - (3*I)*kx*kyi*kyr^2*mf*pr*s + kx*kyr^3*mf*pr*s + (2*I)*kx*kyi^2*kyr*pr^2*s - 2*kx*kyi*kyr^2*pr^2*s - kx*kyi^2*mf*pi*s^2 - kx*kyr^2*mf*pi*s^2 - kx*kyi^2*pi^2*s^2 - kx*kyr^2*pi^2*s^2 - I*kx*kyi^2*mf*pr*s^2 - I*kx*kyr^2*mf*pr*s^2 - I*kx*kyi^2*pi*pr*s^2 - I*kx*kyr^2*pi*pr*s^2 - kx*kyi*mf*pi*s^3 - I*kx*kyr*mf*pi*s^3 - I*kx*kyi*mf*pr*s^3 + kx*kyr*mf*pr*s^3) - Fyc*kx*(kx*kyi^4*mf*pi - (2*I)*kx*kyi^3*kyr*mf*pi - (2*I)*kx*kyi*kyr^3*mf*pi - kx*kyr^4*mf*pi + kx*kyi^4*pi^2 - (2*I)*kx*kyi^3*kyr*pi^2 - (2*I)*kx*kyi*kyr^3*pi^2 - kx*kyr^4*pi^2 - I*kx*kyi^4*mf*pr - 2*kx*kyi^3*kyr*mf*pr - 2*kx*kyi*kyr^3*mf*pr + I*kx*kyr^4*mf*pr - I*kx*kyi^4*pi*pr - 2*kx*kyi^3*kyr*pi*pr - 2*kx*kyi*kyr^3*pi*pr + I*kx*kyr^4*pi*pr + kx*kyi^3*mf*pi*s - (3*I)*kx*kyi^2*kyr*mf*pi*s - 3*kx*kyi*kyr^2*mf*pi*s + I*kx*kyr^3*mf*pi*s - (2*I)*kx*kyi^2*kyr*pi^2*s - 2*kx*kyi*kyr^2*pi^2*s - I*kx*kyi^3*mf*pr*s - 3*kx*kyi^2*kyr*mf*pr*s + (3*I)*kx*kyi*kyr^2*mf*pr*s + kx*kyr^3*mf*pr*s - (2*I)*kx*kyi^2*kyr*pr^2*s - 2*kx*kyi*kyr^2*pr^2*s - kx*kyi^2*mf*pi*s^2 - kx*kyr^2*mf*pi*s^2 - kx*kyi^2*pi^2*s^2 - kx*kyr^2*pi^2*s^2 + I*kx*kyi^2*mf*pr*s^2 + I*kx*kyr^2*mf*pr*s^2 + I*kx*kyi^2*pi*pr*s^2 + I*kx*kyr^2*pi*pr*s^2 - kx*kyi*mf*pi*s^3 + I*kx*kyr*mf*pi*s^3 + I*kx*kyi*mf*pr*s^3 + kx*kyr*mf*pr*s^3) - Fxc*kx*(I*kyi^5*mf*pi + kyi^4*kyr*mf*pi + (2*I)*kyi^3*kyr^2*mf*pi + 2*kyi^2*kyr^3*mf*pi + I*kyi*kyr^4*mf*pi + kyr^5*mf*pi + I*kyi^5*pi^2 + kyi^4*kyr*pi^2 + (2*I)*kyi^3*kyr^2*pi^2 + 2*kyi^2*kyr^3*pi^2 + I*kyi*kyr^4*pi^2 + kyr^5*pi^2 + kyi^5*mf*pr - I*kyi^4*kyr*mf*pr + 2*kyi^3*kyr^2*mf*pr - (2*I)*kyi^2*kyr^3*mf*pr + kyi*kyr^4*mf*pr - I*kyr^5*mf*pr + kyi^5*pi*pr - I*kyi^4*kyr*pi*pr + 2*kyi^3*kyr^2*pi*pr - (2*I)*kyi^2*kyr^3*pi*pr + kyi*kyr^4*pi*pr - I*kyr^5*pi*pr + I*kyi^4*mf*pi*s + 2*kyi^3*kyr*mf*pi*s + 2*kyi*kyr^3*mf*pi*s - I*kyr^4*mf*pi*s + 2*kyi^3*kyr*pi^2*s + 2*kyi*kyr^3*pi^2*s + kyi^4*mf*pr*s - (2*I)*kyi^3*kyr*mf*pr*s - (2*I)*kyi*kyr^3*mf*pr*s - kyr^4*mf*pr*s + 2*kyi^3*kyr*pr^2*s + 2*kyi*kyr^3*pr^2*s - I*kyi^3*mf*pi*s^2 + kyi^2*kyr*mf*pi*s^2 - I*kyi*kyr^2*mf*pi*s^2 + kyr^3*mf*pi*s^2 - I*kyi^3*pi^2*s^2 + kyi^2*kyr*pi^2*s^2 - I*kyi*kyr^2*pi^2*s^2 + kyr^3*pi^2*s^2 - kyi^3*mf*pr*s^2 - I*kyi^2*kyr*mf*pr*s^2 - kyi*kyr^2*mf*pr*s^2 - I*kyr^3*mf*pr*s^2 - kyi^3*pi*pr*s^2 - I*kyi^2*kyr*pi*pr*s^2 - kyi*kyr^2*pi*pr*s^2 - I*kyr^3*pi*pr*s^2 - I*kyi^2*mf*pi*s^3 - I*kyr^2*mf*pi*s^3 - kyi^2*mf*pr*s^3 - kyr^2*mf*pr*s^3) - Fxm*kx*(I*kyi^5*mf*pi - kyi^4*kyr*mf*pi + (2*I)*kyi^3*kyr^2*mf*pi - 2*kyi^2*kyr^3*mf*pi + I*kyi*kyr^4*mf*pi - kyr^5*mf*pi + I*kyi^5*pi^2 - kyi^4*kyr*pi^2 + (2*I)*kyi^3*kyr^2*pi^2 - 2*kyi^2*kyr^3*pi^2 + I*kyi*kyr^4*pi^2 - kyr^5*pi^2 - kyi^5*mf*pr - I*kyi^4*kyr*mf*pr - 2*kyi^3*kyr^2*mf*pr - (2*I)*kyi^2*kyr^3*mf*pr - kyi*kyr^4*mf*pr - I*kyr^5*mf*pr - kyi^5*pi*pr - I*kyi^4*kyr*pi*pr - 2*kyi^3*kyr^2*pi*pr - (2*I)*kyi^2*kyr^3*pi*pr - kyi*kyr^4*pi*pr - I*kyr^5*pi*pr + I*kyi^4*mf*pi*s - 2*kyi^3*kyr*mf*pi*s - 2*kyi*kyr^3*mf*pi*s - I*kyr^4*mf*pi*s - 2*kyi^3*kyr*pi^2*s - 2*kyi*kyr^3*pi^2*s - kyi^4*mf*pr*s - (2*I)*kyi^3*kyr*mf*pr*s - (2*I)*kyi*kyr^3*mf*pr*s + kyr^4*mf*pr*s - 2*kyi^3*kyr*pr^2*s - 2*kyi*kyr^3*pr^2*s - I*kyi^3*mf*pi*s^2 - kyi^2*kyr*mf*pi*s^2 - I*kyi*kyr^2*mf*pi*s^2 - kyr^3*mf*pi*s^2 - I*kyi^3*pi^2*s^2 - kyi^2*kyr*pi^2*s^2 - I*kyi*kyr^2*pi^2*s^2 - kyr^3*pi^2*s^2 + kyi^3*mf*pr*s^2 - I*kyi^2*kyr*mf*pr*s^2 + kyi*kyr^2*mf*pr*s^2 - I*kyr^3*mf*pr*s^2 + kyi^3*pi*pr*s^2 - I*kyi^2*kyr*pi*pr*s^2 + kyi*kyr^2*pi*pr*s^2 - I*kyr^3*pi*pr*s^2 - I*kyi^2*mf*pi*s^3 - I*kyr^2*mf*pi*s^3 + kyi^2*mf*pr*s^3 + kyr^2*mf*pr*s^3)'
str_Fys_num = str_Fys_num.replace('^', '**')



kx, kyr, kyi = sym.symbols('kx kyr kyi', real=True)
ky = kyr + sym.I*kyi
kyc = kyr - sym.I*kyi

Pr, Pi = sym.symbols('Pr Pi', real=True)
P = Pr + sym.I*Pi
Pc = Pr - sym.I*Pi

mf = sym.symbols('mf', real=True) #material factor, Im(P/chi^*)

Lx, Ly, Fxm, Fxc, Fym, Fyc = sym.symbols('Lx Ly Fxm Fxc Fym Fyc')

s = sym.symbols('s') #the Laplace transform variable

symbol_dict = {'kx':kx, 'kyr':kyr, 'kyi':kyi, 'pr':Pr, 'pi':Pi, 'mf':mf,
               'Lx':Lx, 'Ly':Ly, 'Fxm':Fxm, 'Fxc':Fxc, 'Fym':Fym, 'Fyc':Fyc, 
               's':s, 'I':sym.I}


s_denom = parse_expr(str_denom, local_dict=symbol_dict)
Fxs_num = parse_expr(str_Fxs_num, local_dict=symbol_dict)
Fys_num = parse_expr(str_Fys_num, local_dict=symbol_dict)

math_module = ["mpmath"]

denom_s4_coeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf], s_denom.coeff(s, n=4), modules=math_module)
denom_s2_coeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf], s_denom.coeff(s, n=2), modules=math_module)
denom_s0_coeff_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf], s_denom.coeff(s, n=0), modules=math_module)

Fxs_num_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, s], Fxs_num, modules=math_module)
Fys_num_func = sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, s], Fys_num, modules=math_module)

param_list = [Fxm, Fxc, Fym, Fyc]
Fxs_num_param_coeff_func_list = []
Fys_num_param_coeff_func_list = []
Fxs_num_const_expr = Fxs_num.copy()
Fys_num_const_expr = Fys_num.copy()

for param in param_list:
    Fxs_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fxs_num.coeff(param, n=1), modules=math_module))
    Fys_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fys_num.coeff(param, n=1), modules=math_module))
    Fxs_num_const_expr -= param * Fxs_num.coeff(param, n=1)
    Fys_num_const_expr -= param * Fys_num.coeff(param, n=1)

#append the parts of the numerator not associated with any of the free parameters Fxm, Fxc, Fym, Fyc
Fxs_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fxs_num_const_expr, modules=math_module))
Fys_num_param_coeff_func_list.append(sym.lambdify([kx, kyr, kyi, Pr, Pi, mf, s], Fys_num_const_expr, modules=math_module))



def get_mp_TEy_AsymUPinv_expiky_y(chi, phase, k0, kx):
    ky = mp.sqrt(k0**2-kx**2)
    kyr = mp.re(ky); kyi = mp.im(ky)
    
    #print('kyr', kyr, 'kyi', kyi)
    Pr = mp.re(phase); Pi = mp.im(phase)
    mf = mp.im(phase/np.conj(chi))
    #print('material factor', mf)
    
    #find poles of Laplace transform
    quad_a = denom_s4_coeff_func(kx, kyr, kyi, Pr, Pi, mf)
    quad_b = denom_s2_coeff_func(kx, kyr, kyi, Pr, Pi, mf)
    quad_c = denom_s0_coeff_func(kx, kyr, kyi, Pr, Pi, mf)
    
    Delta = quad_b**2 - 4*quad_a*quad_c
    if mp.im(Delta)==0 and mp.re(Delta)<0:
        sqrtDelta = 1j*mp.sqrt(-Delta)
    else:
        sqrtDelta = mp.sqrt(Delta)
    
    sqr_plus = (-quad_b + sqrtDelta) / 2 / quad_a
    sqr_minus = (-quad_b - sqrtDelta) / 2 / quad_a
    
    pole_plus = mp.sqrt(sqr_plus)
    pole_minus = mp.sqrt(sqr_minus)
    
    #print('pole_plus', pole_plus)
    #print('pole_minus', pole_minus)
    if mp.fabs(mp.re(pole_plus))<1e3*mp.eps or mp.fabs(mp.re(pole_minus))<1e3*mp.eps:
        raise ValueError('nan encountered in poles, L2 invertibility lost')

    pole_list = [pole_plus, -pole_plus, pole_minus, -pole_minus]
    #now go through the poles to set up the linear system for solving for the free params
    #free params are set such that exponentially growing poles as y->\infty have residue 0

    negRe_pole_flag = np.zeros(len(pole_list), dtype=np.bool) #flag[i]=True when the pole[i] has negative Re part and is actually involved in the L2 inverse
    posRe_pole_count = 0
    
    param_mat = mp.matrix(4)
    param_b = mp.matrix(4,1)
    for pole_ind, pole in enumerate(pole_list):
        if mp.re(pole)<0:
            negRe_pole_flag[pole_ind] = True
        else:
            if posRe_pole_count == 2:
                raise ValueError('number of positive poles larger than 2, insufficient # of free parameters')
            
            for i in range(4):
                param_mat[2*posRe_pole_count,i] = Fxs_num_param_coeff_func_list[i](kx, kyr, kyi, Pr, Pi, mf, pole)
                param_mat[2*posRe_pole_count+1,i] = Fys_num_param_coeff_func_list[i](kx, kyr, kyi, Pr, Pi, mf, pole)

            param_b[2*posRe_pole_count] = -Fxs_num_param_coeff_func_list[-1](kx, kyr, kyi, Pr, Pi, mf, pole)
            param_b[2*posRe_pole_count+1] = -Fys_num_param_coeff_func_list[-1](kx, kyr, kyi, Pr, Pi, mf, pole)
            
            posRe_pole_count += 1
    
    if posRe_pole_count<2:
        raise ValueError('number of positive poles less than 2, too much freedom from free params')

    # solve for parameter values that will cancel out positive Re pole residues
    try:
        L2_param_vec = mp.lu_solve(param_mat, param_b)
    except ZeroDivisionError:
        #print('Encountered singular matrix at kx', kx, 'use numpy pseudoinverse instead')
        """
        param_U, param_S, param_V = mp.svd(param_mat)
        print('check mpmath svd result', mp.chop(param_mat - param_U * mp.diag(param_S) * param_V))
        param_Sinv = param_S.copy() #get pseudo-inverse SVD diagonal
        param_Sinv[2] = param_Sinv[3] = mp.zero
        param_Sinv[0] = 1.0 / param_Sinv[0]
        param_Sinv[1] = 1.0 / param_Sinv[1]
        
        L2_param_vec = param_U * mp.diag(param_Sinv) * param_V * param_b
        """
        fp_param_mat = np.array(mpmath.fp.matrix(param_mat).tolist())
        fp_param_b = np.array(mpmath.fp.matrix(param_b).tolist()).flatten()
        L2_param_vec = la.pinv(fp_param_mat) @ fp_param_b
        
    Fxm = L2_param_vec[0]; Fxc = L2_param_vec[1]; Fym = L2_param_vec[2]; Fyc = L2_param_vec[3]

    # calculate all the residues of the negative Re poles
    negRe_pole_list = []
    Fxs_res_list = []
    Fys_res_list = []
    
    for pole_ind, pole in enumerate(pole_list):
        if not negRe_pole_flag[pole_ind]:
            continue
        
        negRe_pole_list.append(pole)
        
        res_denom = quad_a
        for i in range(len(pole_list)):
            if i==pole_ind:
                continue
            res_denom *= (pole-pole_list[i])
        
        Fxs_res_list.append(Fxs_num_func(kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, pole) / res_denom)
        Fys_res_list.append(Fys_num_func(kx, kyr, kyi, Pr, Pi, mf, Fxm, Fxc, Fym, Fyc, pole) / res_denom)
    
    return negRe_pole_list, Fxs_res_list, Fys_res_list


def mp_TEy_bound_integrand(d, chi, phase, k0, kx, toDouble=False):
    """
    calculates the kx integrand 
    for the finite bandwidth TM LDOS bound near a half space
    """
    
    PA = mp.sqrt(mp.re(phase*mp.conj(phase)))
    ky = mp.sqrt(k0**2-kx**2)
    kyi = mp.im(ky)
    
    r_list, Rx_list, Ry_list = get_mp_TEy_AsymUPinv_expiky_y(chi, phase, k0, kx)
    
    S2AsymUPinvS1 = mp.mpc(0)
    S1AsymUPinvS1 = mp.mpc(0)
    for i in range(len(r_list)):
        r = r_list[i]
        Rx = Rx_list[i]
        Ry = Ry_list[i]
        S2AsymUPinvS1 += -(kx*Rx + (kx**2/ky)*Ry) / (r + 1j*ky)
        S1AsymUPinvS1 += (kx*Rx - (kx**2/mp.conj(ky))*Ry) / (r - 1j*mp.conj(ky))
    
    S2AsymUPinvS1 = (mp.conj(phase)/k0)*mp.exp(2j*ky*d)*S2AsymUPinvS1
    S1AsymUPinvS1 = (PA/mp.fabs(k0)) * mp.exp(-2*kyi*d) * S1AsymUPinvS1
    #print('S2AsymUPinvS1', S2AsymUPinvS1)
    #print('S1AsymUPinvS1', S1AsymUPinvS1)
    
    #there is an additional (ZJ_0^2/32/pi) prefactor for getting the enhancement bounds later
    k0r = mp.re(k0); k0i = mp.im(k0)
    #wtil = k0r + 1j*k0i
    Ntil = k0r/(k0r**2 + k0i**2)
    integrand = mp.re(S2AsymUPinvS1/(k0*Ntil) + S1AsymUPinvS1/(mp.sqrt(k0r**2 + k0i**2)*Ntil))
    #print('at kx=', kx, 'the integrand value is', integrand)
    if toDouble:
        return np.double(integrand)
    else:
        return integrand


def np_TEy_halfspace_fixed_phase_bound(d, chi, phase, k0, tol=1e-4):
    
    #evaluate the integral in 3 parts: the traveling regime, the lightline regime, the evanescent regime

    k0r = np.real(k0); k0i = np.imag(k0)
    integrand = lambda kx: mp_TEy_bound_integrand(d, chi, phase, k0, kx, toDouble=True)
    mp_integrand = lambda kx: mp_TEy_bound_integrand(d, chi, phase, k0, kx)
    
    traveling_endpoint = k0r
    #there is generally a peak of the integrand closely after the lightline k0r, find the peak to use as an intermediate point for evaluating integral
    delta_lightline = 10*k0i
    lightline_endpoint = traveling_endpoint + delta_lightline
    probe_integrand = integrand(lightline_endpoint)
    while True:
        lightline_endpoint += delta_lightline
        t = integrand(lightline_endpoint)
        if t < probe_integrand:
            break
        probe_integrand = t
        delta_lightline *= 2
    #print('lightline_endpoint', lightline_endpoint, (lightline_endpoint-traveling_endpoint)/k0i)
    evan_endpoint = lightline_endpoint + max(2, 1.0/(k0r*d)) * k0r
    
    #traveling_integral, abserr = np_quad(integrand, 0, traveling_endpoint, epsrel=tol)
    #print('traveling integral', traveling_integral, 'relative error', abserr / traveling_integral)
    mp_traveling_integral = mp.quad(mp_integrand, [0, traveling_endpoint])
    #print('compare traveling integral np and mpmath', traveling_integral, mp_traveling_integral)
    
    
    #lightline_integral, abserr = np_quad(integrand, traveling_endpoint, lightline_endpoint, epsrel=tol)
    #print('lightline integral', lightline_integral, 'relative error', abserr / lightline_integral)
    mp_lightline_integral = mp.quad(mp_integrand, [traveling_endpoint, lightline_endpoint])
    #print('compare lightline integral np and mpmath', lightline_integral, mp_lightline_integral)
    
    
    #evan_integral, abserr = np_quad(integrand, lightline_endpoint, evan_endpoint, epsrel=tol)
    #print('evanescent integral', evan_integral, 'relative error', abserr / evan_integral)
    mp_evan_integral = mp.quad(mp_integrand, [lightline_endpoint, evan_endpoint])
    #print('compare evan integral np and mpmath', evan_integral, mp_evan_integral)

    
    integral = mp_traveling_integral + mp_lightline_integral + mp_evan_integral
    
    delta_kx = 10*k0r
    while True: #keep on integrating the deep evanescent tail until desired accuracy is reached
        #delta_integral, abserr = np_quad(integrand, evan_endpoint, evan_endpoint+delta_kx, epsrel=tol)
        mp_delta_integral = mp.quad(integrand, [evan_endpoint, evan_endpoint+delta_kx])
        integral += mp_delta_integral
        if abs(mp_delta_integral / integral) < 1e-5:
            break
        evan_endpoint += delta_kx
    
    integral *= 2

    #prefactor is (Z*J_0^2/32/np.pi) / (Pvac=np.pi*Z*J_0^2/4/wavelength0)
    wvlgth0 = 2*np.pi / k0r
    rhovac0 = 2*np.pi/16
    rho0 = (k0r**2 + k0i**2)/k0r/(8*np.pi)*np.arctan2(k0r, k0i)
    #return 1 + (wvlgth0/8/np.pi**2) * integral #the 1 is the constant vacuum contribution
    return float((wvlgth0/8/np.pi**2) * rhovac0/rho0 * integral) #dont include the constant vacuum contribution



def mp_TEy_halfspace_fixed_phase_bound(d, chi, phase, k0):
    """
    evaluate the integral in 3 parts: the traveling regime, the lightline regime, the evanescent regime
    if necessary in future adjust so large bandwidths don't raise error
    """
    k0r = mp.re(k0); k0i = mp.im(k0)
    integrand = lambda kx: mp_TEy_bound_integrand(d, chi, phase, k0, kx)
    
    end_kx = 3*k0r+50*k0i
    integral = mp.quad(integrand, [0, k0r-50*k0i, k0r+50*k0i, end_kx])
    delta_kx = 10*k0r
    while True: #keep on integrating the deep evanescent tail until desired accuracy is reached
        delta_integral = mp.quad(integrand, [end_kx, end_kx+delta_kx])
        integral += delta_integral
        if mp.fabs(delta_integral / integral) < 1e-5:
            break
        end_kx += delta_kx
        
    #extra factor of 2 for the symmetric integral
    integral *= 2
    
    #prefactor
    wvlgth0 = 2*mp.pi / k0r
    #return (wvlgth0/8/mp.pi**2) * integral #the 1 is the constant vacuum contribution
    rhovac0 = 2*mp.pi/16
    rho0 = (k0r**2 + k0i**2)/k0r/(8*mp.pi)*mp.atan2(k0r, k0i)
    return (wvlgth0/8/mp.pi**2) * rhovac0/rho0 * integral #dont include the constant vacuum contribution


def check_AsymUP_full_space_mineigs(chi, k0, p):
    """
    returns the minimum transverse eigenvalue and the longitudinal eigenvalue for
    AsymUP + Im(p/chi^*) + Asym(p^* G) over the entire space
    """
    p_re = np.real(p); p_im = np.imag(p)
    rhoUP_l = np.imag(p/np.conj(chi)) + p_im
    
    #to find min(rho_t), look at the 4 critical points k^2=0, k^2->infty and the two internal stationary points
    #calculate the eigenvalues at the two finite k stationary points
    A_re = np.real(k0**2); A_im = np.real(k0**2)
    A_norm = np.sqrt(A_re**2 + A_im**2)
    p_norm = np.sqrt(p_re**2 + p_im**2)
    usqr_coeff = p_im*A_re - p_re*A_im
    
    if usqr_coeff==0.0:
        #u = k^2 = A_re is only finite k stationary point
        rhoG_t = p_im * (A_norm/A_im)**2
    else:
        u_plus = (p_im*(A_norm**2) + A_im*A_norm*p_norm) / (p_im*A_re - p_re*A_im)
        u_minus = (p_im*(A_norm**2) - A_im*A_norm*p_norm) / (p_im*A_re - p_re*A_im)
        if u_plus>=0:
            rhoG_t_plus = (p_re*A_im*u_plus + p_im*(A_norm**2 - A_re*u_plus)) / ((u_plus-A_re)**2 + A_im**2)
        else:
            rhoG_t_plus = np.inf #this stationary point outside definition for k^2
        if u_minus>=0:
            rhoG_t_minus = (p_re*A_im*u_minus + p_im*(A_norm**2 - A_re*u_minus)) / ((u_minus-A_re)**2 + A_im**2)
        else:
            rhoG_t_minus = np.inf #this stationary point outside definition for k^2
        rhoG_t = min(rhoG_t_plus, rhoG_t_minus)
        
    rhoG_t = min(p_im, 0, rhoG_t) #rhoG_t(k=0) = p_im, rhoG_t(k->\infty)=0
    rhoUP_t = np.imag(p/np.conj(chi)) + rhoG_t
    return rhoUP_t, rhoUP_l


def np_TEy_halfspace_bound(d, chi, k0):
    """
    calculate the tightest possible dual bound for a TE y-polarized dipole near a half-space design domain
    given the complex global energy conservation constraints
    """
    theta_boundfunc = lambda angle: np_TEy_halfspace_fixed_phase_bound(d, chi, np.exp(1j*angle), k0)
    
    #phase angle -pi < theta < pi; find upper and lower limits on phase angle theta
    delta_theta = np.imag(k0) / np.real(k0) / 2
    theta_r = 1.3 * delta_theta
    while True:
        try:
            theta_boundfunc(theta_r)
            break
        except:
            theta_r /= 2.0
    
    probe_bound = theta_boundfunc(theta_r)
    
    while True: #find upper bound on optimal phase angle theta
        reduced_stepsize = False
        while True:
            if theta_r+delta_theta > np.pi:
                delta_theta = 2*(np.pi-theta_r)/3.0
                reduced_stepsize = True
                continue
            try:
                t = theta_boundfunc(theta_r+delta_theta)
                break
            except ValueError: #inverse of AsymUP acting on S1 is not in L2
                delta_theta /= 2
                reduced_stepsize = True
            
        theta_r += delta_theta
        if not reduced_stepsize:
            delta_theta *= 2
        if t>probe_bound:
            break
        probe_bound = t
    
    theta_l = 2*theta_r/3
    probe_bound = theta_boundfunc(theta_l)
    delta_theta = theta_r / 2.0
    onBoundary=False
    while True: #find lower bound on optimal phase angle theta
        reduced_stepsize = False
        while True:
            if theta_l - delta_theta < -np.pi:
                delta_theta = 2*(theta_l+np.pi)/3.0
                reduced_stepsize = True
                continue
            try:
                t = theta_boundfunc(theta_l-delta_theta)
                break
            except ValueError: #inverse of AsymUP acting on S1 is not in L2
                delta_theta /= 2
                reduced_stepsize = True
        
        theta_l -= delta_theta
        #print('theta_l', theta_l, 'probe_bound', probe_bound, 't', t)
        if not reduced_stepsize:
            delta_theta *= 2
        if t>probe_bound:
            break
        if reduced_stepsize and delta_theta < max(1e-10, abs(theta_l)*1e-4):
            onBoundary=True
            break
        probe_bound = t

    if onBoundary:
        theta_opt = theta_l
        bound = theta_boundfunc(theta_opt)
    else:
        opt = minimize_scalar(theta_boundfunc, bounds=(theta_l, theta_r), method='bounded', options={'xatol':min(1e-3, (theta_r-theta_l)/100)})

        theta_opt = opt.x
        bound = opt.fun

    print('theta_l', theta_l, 'tan(theta_l)', 'theta_r', theta_r)
    rhoUP_t_theta_r, rhoUP_l_theta_r = check_AsymUP_full_space_mineigs(np.complex(chi), np.complex(k0), np.exp(1j*theta_r))
    rhoUP_t_theta_l, rhoUP_l_theta_l = check_AsymUP_full_space_mineigs(np.complex(chi), np.complex(k0), np.exp(1j*theta_l))
    rhoUP_t_theta_opt, rhoUP_l_theta_opt = check_AsymUP_full_space_mineigs(np.complex(chi), np.complex(k0), np.exp(1j*theta_opt))
    print('checking the eigenvalues for AsymUP of the whole space')
    print('at theta_r, the mineig for transverse and longitudinal', rhoUP_t_theta_r, rhoUP_l_theta_r)
    print('at theta_l, the mineig for transverse and longitudinal', rhoUP_t_theta_l, rhoUP_l_theta_l)
    print('at theta_opt, the mineig for transverse and longitudinal', rhoUP_t_theta_opt, rhoUP_l_theta_opt)
    
    return bound, theta_opt



"""
plotting functions
"""

def plot_TEy_bound_integrand(d, chi, phase, k0, kx_min, kx_max, kx_num):
    kx_list = mp.linspace(kx_min, kx_max, kx_num)
    integrand_list = []
    for i,kx in enumerate(kx_list):
        try:
            integrand = mp_TEy_bound_integrand(d, chi, phase, k0, kx)
        except:
            integrand = mp.zero
        integrand_list.append(integrand)
    
    plt.plot(kx_list, integrand_list)
    

def plot_TEy_bound_vary_phase(d, chi, k0, theta_l, theta_r, theta_num):
    theta_list = np.linspace(theta_l, theta_r, theta_num)
    bound_list = np.zeros_like(theta_list)
    
    for i,theta in enumerate(theta_list):
        print('i', i, 'theta', theta)
        bound_list[i] = np_TEy_halfspace_fixed_phase_bound(d, chi, np.exp(1j*theta), k0)
    
    plt.plot(theta_list, bound_list)
