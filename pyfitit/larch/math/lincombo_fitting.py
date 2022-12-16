#!/usr/bin/env python
"""
linear combination fitting
"""
import os
import sys
import time
import json
import copy

from itertools import combinations
from collections import OrderedDict
from glob import glob

import numpy as np
from numpy.random import randint
import lmfit
from .. import Group
from .utils import interp, index_of, etok


def get_arrays(group, arrayname, xname='energy'):
    y = None
    if arrayname == 'chik':
        x = getattr(group, 'k', None)
        y = getattr(group, 'chi', None)
    else:
        x = getattr(group, xname, None)
        y = getattr(group, arrayname, None)
    return x, y

def get_label(group):
    label = None
    for attr in ('filename', 'label', 'groupname', '__name__'):
        _lab = getattr(group, attr, None)
        if _lab is not None:
            label = _lab
            break
    if label is None:
        label = hex(id(group))
    return label


def groups2matrix(groups, yname='norm', xname='energy', xmin=-np.inf, xmax=np.inf,
                  interp_kind='cubic'):
    """extract an array from a list of groups and construct a uniform 2d matrix
    ready for linear analysis

    Argumments
    ----------
    groups       list of groups, assumed to have similar naming conventions
    yname        name of y-arrays to convert to matrix ['norm']
    xname        name of x-array to use ['energy']
    xmin         min x value [-inf]
    xmax         max x value [+inf]
    interp_kind  kind argument for interpolation ['cubic']

    Returns
    -------
    xdat, ydat  where xdat has shape (nx,) and ydat has shape (nx, ngroups)
    """
    # get arrays from first group
    kweight = 0
    if yname.startswith('chi'):
        xname = 'k'
        if len(yname) > 3:
            kweight = int(yname[3:])
        yname = 'chi'
        e0 = getattr(groups[0], 'e0', -1.)
        if e0 < 0:
            raise ValueError("cannot get chi data")

    xdat, ydat = get_arrays(groups[0], yname, xname=xname)
    if xdat is None or ydat is None:
        raise ValueError("cannot get arrays for arrayname='%s'" % yname)

    imin, imax = None, None
    if xmin is not None:
        if xname == 'k':
            if xmin > e0:
                xmin = etok(xmin-e0)
            else:
                xmin = 0.0

        imin = index_of(xdat, xmin)
    if xmax is not None:
        if xname == 'k':
            if xmax > e0:
                xmax = etok(xmax-e0)
            else:
                xmax = max(xdat)
        imax = index_of(xdat, xmax) + 1

    xsel = slice(imin, imax)
    xdat = xdat[xsel]
    ydat = ydat[xsel]
    if xname == 'k' and kweight > 0:
        ydat = ydat * xdat**kweight
    ydat = [ydat]
    for g in groups[1:]:
        x, y = get_arrays(g, yname, xname=xname)
        if xname == 'k' and kweight > 0:
            y = y * x**kweight
        ydat.append(interp(x, y, xdat, kind=interp_kind))
    return xdat, np.array(ydat)


def lincombo_fit(group, components, weights=None, minvals=None,
                 maxvals=None, arrayname='norm', xmin=-np.inf, xmax=np.inf,
                 sum_to_one=True, vary_e0=False, max_ncomps=None):

    """perform linear combination fitting for a group

    Arguments
    ---------
    group       Group to be fitted
    components  List of groups to use as components (see Note 1)
    weights     array of starting  weights (see Note 2)
    minvals     array of min weights (or None to mean -inf)
    maxvals     array of max weights (or None to mean +inf)
    arrayname   string of array name to be fit (see Note 3) ['norm']
    xmin        x-value for start of fit range [-inf]
    xmax        x-value for end of fit range [+inf]
    sum_to_one  bool, whether to force weights to sum to 1.0 [True]
    vary_e0     bool, whether to vary e0 for data in fit [False]

    Returns
    -------
    group with resulting weights and fit statistics

    Notes
    -----
     1.  The names of Group members for the components must match those of the
         group to be fitted.
     2.  use `None` to use basic linear algebra solution.
     3.  arrayname is expected to be one of `norm`, `mu`, `dmude`, or `chi`.
         It can be some other name but such named arrays should exist for all
         components and groups.
    """

    # first, gather components
    ncomps = len(components)
    allgroups = [group]
    allgroups.extend(components)
    xdat, yall = groups2matrix(allgroups, yname=arrayname,
                               xname='energy', xmin=xmin, xmax=xmax)

    ydat   = yall[0, :]
    ycomps = yall[1:, :].transpose()

    # second use unconstrained linear algebra to estimate weights
    ls_out = np.linalg.lstsq(ycomps, ydat, rcond=-1)
    ls_vals = ls_out[0]
    # third use lmfit, imposing bounds and sum_to_one constraint
    if weights in (None, [None]*ncomps):
        weights = ls_vals
    if minvals in (None, [None]*ncomps):
        minvals = -np.inf * np.ones(ncomps)
    if maxvals in (None, [None]*ncomps):
        maxvals = np.inf * np.ones(ncomps)

    def lincombo_resid(params, xdata, ydata, ycomps):
        npts, ncomps = ycomps.shape
        if params['e0_shift'].vary:
            y = interp(xdata, ydata, xdata+params['e0_shift'].value, kind='cubic')
        else:
            y = ydata*1.0
        resid = -y
        for i in range(ncomps):
            resid += ycomps[:, i] * params['c%i' % i].value
        return resid

    params = lmfit.Parameters()
    e0_val = 0.01 if vary_e0 else 0.
    params.add('e0_shift', value=e0_val, vary=vary_e0)
    for i in range(ncomps):
        params.add('c%i' % i, value=weights[i], min=minvals[i], max=maxvals[i])

    if sum_to_one:
        expr = ['1'] + ['c%i' % i for i in range(ncomps-1)]
        params['c%i' % (ncomps-1)].expr = '-'.join(expr)

    expr = ['c%i' % i for i in range(ncomps)]
    params.add('total', expr='+'.join(expr))

    result = lmfit.minimize(lincombo_resid, params, args=(xdat, ydat, ycomps))

    # gather results
    weights, weights_lstsq = OrderedDict(), OrderedDict()
    params, fcomps = OrderedDict(), OrderedDict()
    params['e0_shift'] = copy.deepcopy(result.params['e0_shift'])
    for i in range(ncomps):
        label = get_label(components[i])
        weights[label] = result.params['c%i' % i].value
        params[label] = copy.deepcopy(result.params['c%i' % i])
        weights_lstsq[label] = ls_vals[i]
        fcomps[label] = ycomps[:, i] * result.params['c%i' % i].value


    if 'total' in result.params:
        params['total'] = copy.deepcopy(result.params['total'])

    npts, ncomps = ycomps.shape
    yfit = np.zeros(npts)
    for i in range(ncomps):
        yfit += ycomps[:, i] * result.params['c%i' % i].value
    if params['e0_shift'].vary:
        yfit = interp(xdat+params['e0_shift'].value, yfit, xdat, kind='cubic')
    rfactor = ((ydat-yfit)**2).sum() / (ydat**2).sum()
    return Group(result=result, chisqr=result.chisqr, redchi=result.redchi,
                 params=params, weights=weights, weights_lstsq=weights_lstsq,
                 xdata=xdat, ydata=ydat, yfit=yfit, ycomps=fcomps,
                 arrayname=arrayname, rfactor=rfactor,
                 xmin=xmin, xmax=xmax)

def lincombo_fitall(group, components, weights=None, minvals=None, maxvals=None,
                    arrayname='norm', xmin=-np.inf, xmax=np.inf,
                    max_ncomps=None, sum_to_one=True, vary_e0=False,
                    min_weight=0.0005, max_output=16):
    """perform linear combination fittings for a group with all combinations
    of 2 or more of the components given

    Arguments
    ---------
      group       Group to be fitted
      components  List of groups to use as components (see Note 1)
      weights     array of starting  weights (or None to use basic linear alg solution)
      minvals     array of min weights (or None to mean -inf)
      maxvals     array of max weights (or None to mean +inf)
      arrayname   string of array name to be fit (see Note 2)
      xmin        x-value for start of fit range [-inf]
      xmax        x-value for end of fit range [+inf]
      sum_to_one  bool, whether to force weights to sum to 1.0 [True]
      max_ncomps  int or None: max number of components to use [None -> all]
      vary_e0     bool, whether to vary e0 for data in fit [False]
      min_weight  float, minimum weight for each component to save result [0.0005]
      max_output  int, max number of outputs, sorted by reduced chi-square [16]
    Returns
    -------
     list of groups with resulting weights and fit statistics, ordered by
         reduced chi-square (best first)

    Notes
    -----
     1.  The names of Group members for the components must match those of the
         group to be fitted.
     2.  arrayname can be one of `norm` or `dmude`
    """

    ncomps = len(components)

    # here we save the inputs weights and bounds for each component by name
    # so they can be imposed for the individual fits
    _save = {}

    if weights in (None, [None]*ncomps):
        weights = [None]*ncomps
    if minvals in (None, [None]*ncomps):
        minvals = -np.inf * np.ones(ncomps)
    if maxvals in (None, [None]*ncomps):
        maxvals = np.inf * np.ones(ncomps)

    for i in range(ncomps):
        _save[get_label(components[i])] = (weights[i], minvals[i], maxvals[i])

    if max_ncomps is None:
        max_ncomps = ncomps
    elif max_ncomps > 0:
        max_ncomps = int(min(max_ncomps, ncomps))
    out = []
    nrejected = 0
    comps_kept = []
    for nx in range(2, int(max_ncomps)+1):
        for comps in combinations(components, nx):
            labs = [get_label(c) for c in comps]
            _wts = [1.0/nx for lab in labs]
            _min = [_save[lab][1] for lab in labs]
            _max = [_save[lab][2] for lab in labs]

            ret = lincombo_fit(group, comps, weights=_wts,
                               arrayname=arrayname, minvals=_min,
                               maxvals=_max, xmin=xmin, xmax=xmax,
                               sum_to_one=sum_to_one, vary_e0=vary_e0)

            _sig_comps = []
            for key, wt in ret.weights.items():
                if wt > min_weight:
                    _sig_comps.append(key)
            _sig_comps.sort()
            if _sig_comps not in comps_kept:
                comps_kept.append(_sig_comps)
                out.append(ret)
            else:
                nrejected += 1

    # sort outputs by reduced chi-square
    # print("lin combo : ", len(out), nrejected, max_output)
    return sorted(out, key=lambda x: x.redchi)[:max_output]
