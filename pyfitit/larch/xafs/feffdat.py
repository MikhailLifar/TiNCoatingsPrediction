#!/usr/bin/env python
"""
feffdat  provides the following function related to
reading and dealing with Feff.data files in larch:

  path1 = read_feffdat('feffNNNN.dat')

returns a Feff Group -- a special variation of a Group -- for
the path represented by the feffNNNN.dat

  group  = ff2chi(paths)

creates a group that contains the chi(k) for the sum of paths.
"""
import os
import numpy as np
from copy import deepcopy
from scipy.interpolate import UnivariateSpline
from lmfit import Parameters, Parameter
from lmfit.printfuncs import gformat

# from xraydb import atomic_mass, atomic_symbol

from pyfitit.larch import Group, isNamedClass
# from pyfitit.larch.utils.strutils import fix_varname, b32hash
from pyfitit.larch.fitting import group2params, isParameter, param_value

from .xafsutils import ETOK, ktoe, set_xafsGroup
from .sigma2_models import add_sigma2funcs

SMALL_ENERGY = 1.e-6

class FeffDatFile(Group):
    def __init__(self, filename=None,  **kws):
        kwargs = dict(name='feff.dat: %s' % filename)
        kwargs.update(kws)
        Group.__init__(self,  **kwargs)
        if filename is not None:
            self._read(filename)

    def __repr__(self):
        if self.filename is not None:
            return '<Feff.dat File Group: %s>' % self.filename
        return '<Feff.dat File Group (empty)>'

    def __copy__(self):
        return FeffDatFile(filename=self.filename)

    def __deepcopy__(self, memo):
        return FeffDatFile(filename=self.filename)

    @property
    def reff(self): return self.__reff__

    @reff.setter
    def reff(self, val):     pass

    @property
    def nleg(self): return self.__nleg__

    @nleg.setter
    def nleg(self, val):    pass

    @property
    def rmass(self):
        """reduced mass for a path"""
        if self.__rmass is None:
            rmass = 0
            for atsym, iz, ipot, amass, x, y, z in self.geom:
                rmass += 1.0/max(1., amass)
            self.__rmass = 1./rmass
        return self.__rmass

    @rmass.setter
    def rmass(self, val):  pass

    def __setstate__(self, state):
        (self.filename, self.title, self.version, self.shell,
         self.absorber, self.degen, self.__reff__, self.__nleg__,
         self.rnorman, self.edge, self.gam_ch, self.exch, self.vmu, self.vfermi,
         self.vint, self.rs_int, self.potentials, self.geom, self.__rmass,
         self.k, self.real_phc, self.mag_feff, self.pha_feff,
         self.red_fact, self.lam, self.rep, self.pha, self.amp) = state

        self.k = np.array(self.k)
        self.real_phc = np.array(self.real_phc)
        self.mag_feff = np.array(self.mag_feff)
        self.pha_feff = np.array(self.pha_feff)
        self.red_fact = np.array(self.red_fact)
        self.lam = np.array(self.lam)
        self.rep = np.array(self.rep)
        self.pha = np.array(self.pha)
        self.amp = np.array(self.amp)

    def __getstate__(self):
        return (self.filename, self.title, self.version, self.shell,
                self.absorber, self.degen, self.__reff__, self.__nleg__,
                self.rnorman, self.edge, self.gam_ch, self.exch, self.vmu,
                self.vfermi, self.vint, self.rs_int, self.potentials,
                self.geom, self.__rmass, self.k.tolist(),
                self.real_phc.tolist(), self.mag_feff.tolist(),
                self.pha_feff.tolist(), self.red_fact.tolist(),
                self.lam.tolist(), self.rep.tolist(), self.pha.tolist(),
                self.amp.tolist())


    def _read(self, filename):
        try:
            with open(filename, 'r') as fh:
                lines = fh.readlines()
        except:
            print('Error reading file %s ' % filename)
            return
        self.filename = filename
        mode = 'header'
        self.potentials, self.geom = [], []
        data = []
        pcounter = 0
        iline = 0
        for line in lines:
            iline += 1
            line = line[:-1].strip()
            if line.startswith('#'): line = line[1:]
            line = line.strip()
            if iline == 1:
                self.title = line[:64].strip()
                self.version = line[64:].strip()
                continue
            if line.startswith('k') and line.endswith('real[p]@#'):
                mode = 'arrays'
                continue
            elif '----' in line[2:10]:
                mode = 'path'
                continue
            #
            if (mode == 'header' and
                line.startswith('Abs') or line.startswith('Pot')):
                words = line.replace('=', ' ').split()
                ipot, z, rmt, rnm = (0, 0, 0, 0)
                words.pop(0)
                if line.startswith('Pot'):
                    ipot = int(words.pop(0))
                iz = int(words[1])
                rmt = float(words[3])
                rnm = float(words[5])
                if line.startswith('Abs'):
                    self.shell = words[6]
                self.potentials.append((ipot, iz, rmt, rnm))
            elif mode == 'header' and line.startswith('Gam_ch'):
                words  = line.replace('=', ' ').split(' ', 2)
                self.gam_ch = float(words[1])
                self.exch   = words[2]
            elif mode == 'header' and line.startswith('Mu'):
                words  = line.replace('=', ' ').replace('eV', ' ').split()
                self.vmu = float(words[1])
                self.vfermi = ktoe(float(words[3]))
                self.vint = float(words[5])
                self.rs_int= float(words[7])
            elif mode == 'path':
                pcounter += 1
                if pcounter == 1:
                    w = [float(x) for x in line.split()[:5]]
                    self.__nleg__ = int(w.pop(0))
                    self.degen, self.__reff__, self.rnorman, self.edge = w
                elif pcounter > 2:
                    words = line.split()
                    xyz = ["%7.4f" % float(x) for x in words[:3]]
                    ipot = int(words[3])
                    iz   = int(words[4])
                    if len(words) > 5:
                        lab = words[5]
                    else:
                        lab = atomic_symbol(iz)
                    amass = atomic_mass(iz)
                    geom = [lab, iz, ipot, amass] + xyz
                    if len(self.geom) == 0:
                        self.absorber = lab
                    self.geom.append(tuple(geom))
            elif mode == 'arrays':
                d = np.array([float(x) for x in line.split()])
                if len(d) == 7:
                    data.append(d)
        data = np.array(data).transpose()
        self.k        = data[0]
        self.real_phc = data[1]
        self.mag_feff = data[2]
        self.pha_feff = data[3]
        self.red_fact = data[4]
        self.lam = data[5]
        self.rep = data[6]
        self.pha = data[1] + data[3]
        self.amp = data[2] * data[4]
        self.__rmass = None  # reduced mass of path


PATH_PARS = ('degen', 's02', 'e0', 'ei', 'deltar', 'sigma2', 'third', 'fourth')

class FeffPathGroup(Group):
    def __init__(self, filename='', label='', feffrun='', s02=None, degen=None,
                 e0=None, ei=None, deltar=None, sigma2=None, third=None,
                 fourth=None, use=True, **kws):

        kwargs = dict(filename=filename)
        kwargs.update(kws)
        Group.__init__(self, **kwargs)

        self.filename = filename
        self.feffrun = feffrun
        self.label = label
        self.use = use
        self.params = None
        self.spline_coefs = None
        self.geom  = []
        self.shell = 'K'
        self.absorber = None
        self._feffdat = None

        self.hashkey = 'p000'
        self.k = None
        self.chi = None

        self.__def_degen = 1
        if filename not in ('', None):
            if not os.path.exists(filename):
                raise ValueError(f"Feff Path file '{filename:s}' not found")
            self._feffdat = FeffDatFile(filename=filename)
            self.create_spline_coefs()

            self.geom  = self._feffdat.geom
            self.shell = self._feffdat.shell
            self.absorber = self._feffdat.absorber
            self.__def_degen  = self._feffdat.degen

            self.hashkey = self.__geom2label()
            if self.label in ('', None):
                self.label = self.hashkey

            if feffrun in ('',  None):
                try:
                    dirname, fpfile = os.path.split(filename)
                    parent, folder = os.path.split(dirname)
                    self.feffrun = folder
                except:
                    pass

        self.init_path_params(degen=degen, s02=s02, e0=e0, ei=ei,
                              deltar=deltar, sigma2=sigma2, third=third,
                              fourth=fourth)

    def init_path_params(self, degen=None, s02=None, e0=None, ei=None,
                       deltar=None, sigma2=None, third=None, fourth=None):
        """set inital values/expressions for path parameters for Feff Path"""
        self.degen = self.__def_degen if degen  is None else degen
        self.s02    = 1.0      if s02    is None else s02
        self.e0     = 0.0      if e0     is None else e0
        self.ei     = 0.0      if ei     is None else ei
        self.deltar = 0.0      if deltar is None else deltar
        self.sigma2 = 0.0      if sigma2 is None else sigma2
        self.third  = 0.0      if third  is None else third
        self.fourth = 0.0      if fourth is None else fourth

    def __repr__(self):
        if self.filename is not None:
            return 'feffpath((no_file)'
        return f'feffpath({self.filename})'

    def __getstate__(self):
        _feffdat_state = self._feffdat.__getstate__()
        return (self.filename, self.label, self.feffrun, self.degen,
                self.s02, self.e0, self.ei, self.deltar, self.sigma2,
                self.third, self.fourth, self.use, _feffdat_state)


    def __setstate__(self, state):
        self.params = self.spline_coefs = self.k = self.chi = None
        self.use = True
        if len(state) == 12:  # "use" was added after paths states were being saved
            (self.filename, self.label, self.feffrun, self.degen,
             self.s02, self.e0, self.ei, self.deltar, self.sigma2,
             self.third, self.fourth, _feffdat_state) = state
        elif len(state) == 13:
            (self.filename, self.label, self.feffrun, self.degen,
             self.s02, self.e0, self.ei, self.deltar, self.sigma2,
             self.third, self.fourth, self.use, _feffdat_state) = state


        self._feffdat = FeffDatFile()
        self._feffdat.__setstate__(_feffdat_state)

        self.create_spline_coefs()

        self.geom  = self._feffdat.geom
        self.shell = self._feffdat.shell
        self.absorber = self._feffdat.absorber
        def_degen  = self._feffdat.degen

        self.hashkey = self.__geom2label()
        if self.label in ('', None):
            self.label = self.hashkey


    def __geom2label(self):
        """generate label by hashing path geometry"""
        rep = [self._feffdat.degen, self._feffdat.shell, self.feffrun]
        for atom in self.geom:
            rep.extend(atom)
        rep.append("%7.4f" % self._feffdat.reff)
        s = "|".join([str(i) for i in rep])
        return "p%s" % (b32hash(s)[:9].lower())

    def pathpar_name(self, parname):
        """
        get internal name of lmfit Parameter for a path paramter, using Path's hashkey
        """
        return f'{parname}_{self.hashkey}'

    def __copy__(self):
        newpath = FeffPathGroup()
        newpath.__setstate__(self.__getstate__())
        return newpath

    def __deepcopy__(self, memo):
        newpath = FeffPathGroup()
        newpath.__setstate__(self.__getstate__())
        return newpath


    @property
    def reff(self): return self._feffdat.reff

    @reff.setter
    def reff(self, val):  pass

    @property
    def nleg(self): return self._feffdat.nleg

    @nleg.setter
    def nleg(self, val):     pass

    @property
    def rmass(self): return self._feffdat.rmass

    @rmass.setter
    def rmass(self, val):  pass

    def __repr__(self):
        return f'<FeffPath Group label={self.label:s}, filename={self.filename:s}, use={self.use}>'

    def create_path_params(self, params=None):
        """
        create Path Parameters within the current lmfit.Parameters namespace
        """
        if params is not None:
           self.params = params
        if self.params is None:
            self.params = Parameters()

        if self.params._asteval.symtable.get('sigma2_debye', None) is None:
            add_sigma2funcs(self.params)
        if self.label is None:
            self.label = self.__geom2label()
        self.store_feffdat()
        for pname in PATH_PARS:
            val =  getattr(self, pname)
            attr = 'value'
            if isinstance(val, str):
                attr = 'expr'
            kws =  {'vary': False, attr: val}
            parname = self.pathpar_name(pname)
            self.params.add(parname, **kws)
            self.params[parname].is_pathparam = True

    def create_spline_coefs(self):
        """pre-calculate spline coefficients for feff data"""
        self.spline_coefs = {}
        fdat = self._feffdat
        self.spline_coefs['pha'] = UnivariateSpline(fdat.k, fdat.pha, s=0)
        self.spline_coefs['amp'] = UnivariateSpline(fdat.k, fdat.amp, s=0)
        self.spline_coefs['rep'] = UnivariateSpline(fdat.k, fdat.rep, s=0)
        self.spline_coefs['lam'] = UnivariateSpline(fdat.k, fdat.lam, s=0)

    def store_feffdat(self):
        """stores data about this Feff path in the Parameters
        symbol table for use as `reff` and in sigma2 calcs
        """
        symtab = self.params._asteval.symtable
        symtab['feffpath'] = self._feffdat
        symtab['reff'] = self._feffdat.reff
        symtab['vint'] = self._feffdat.vint
        symtab['vmu']  = self._feffdat.vmu
        symtab['vfermi'] = self._feffdat.vfermi


    def __path_params(self, **kws):
        """evaluate path parameter value.  Returns
        (degen, s02, e0, ei, deltar, sigma2, third, fourth)
        """
        # put 'reff' and '_feffdat' into the symboltable so that
        # they can be used in constraint expressions
        self.store_feffdat()
        if self.params is None:
            self.create_path_params()
        out = []
        for pname in PATH_PARS:
            val = kws.get(pname, None)
            if val is None:
                parname = self.pathpar_name(pname)
                val = self.params[parname]._getval()
            out.append(val)
        return out

    def path_paramvals(self, **kws):
        (deg, s02, e0, ei, delr, ss2, c3, c4) = self.__path_params()
        return dict(degen=deg, s02=s02, e0=e0, ei=ei, deltar=delr,
                    sigma2=ss2, third=c3, fourth=c4)

    def report(self):
        "return  text report of parameters"
        tmpvals = self.__path_params()
        pathpars = {}
        for pname in ('degen', 's02', 'e0', 'deltar',
                      'sigma2', 'third', 'fourth', 'ei'):
            parname = self.pathpar_name(pname)
            if parname in self.params:
                pathpars[pname] = (self.params[parname].value,
                                   self.params[parname].stderr)

        out = [f" = Path '{self.label}' = {self.absorber} {self.shell} Edge",
               f"    feffdat file = {self.filename}, from feff run '{self.feffrun}'"]
        geomlabel  = '    geometry  atom      x        y        z      ipot'
        geomformat = '            %4s      %s, %s, %s  %d'
        out.append(geomlabel)

        for atsym, iz, ipot, amass, x, y, z in self.geom:
            s = geomformat % (atsym, x, y, z, ipot)
            if ipot == 0: s = "%s (absorber)" % s
            out.append(s)

        stderrs = {}
        out.append('     {:7s}= {:s}'.format('reff',
                                              gformat(self._feffdat.reff)))

        for pname in ('degen', 's02', 'e0', 'r',
                      'deltar', 'sigma2', 'third', 'fourth', 'ei'):
            val = strval = getattr(self, pname, 0)
            parname = self.pathpar_name(pname)
            std = None
            if pname == 'r':
                parname = self.pathpar_name('deltar')
                par = self.params.get(parname, None)
                val = par.value + self._feffdat.reff
                strval = 'reff + ' + getattr(self, 'deltar', 0)
                std = par.stderr
            else:
                if pname in pathpars:
                    val, std = pathpars[pname]
                else:
                    par = self.params.get(parname, None)
                    if par is not None:
                        val = par.value
                        std = par.stderr

            if std is None  or std <= 0:
                svalue = gformat(val)
            else:
                svalue = "{:s} +/-{:s}".format(gformat(val), gformat(std))
            if pname == 's02':
                pname = 'n*s02'

            svalue = "     {:7s}= {:s}".format(pname, svalue)
            if isinstance(strval, str):
                svalue = "{:s}  := '{:s}'".format(svalue, strval)

            if val == 0 and pname in ('third', 'fourth', 'ei'):
                continue
            out.append(svalue)
        return '\n'.join(out)

    def calc_chi_from_params(self, params, **kws):
        "calculate chi(k) from Parameters, ParameterGroup, and/or kws for path parameters"
        if isinstance(params, Parameters):
            self.create_path_params(params=params)
        else:
            self.create_path_params(params=group2params(params))
        self._calc_chi(**kws)

    def _calc_chi(self, k=None, kmax=None, kstep=None, degen=None, s02=None,
                 e0=None, ei=None, deltar=None, sigma2=None,
                 third=None, fourth=None, debug=False, interp='cubic', **kws):
        """calculate chi(k) with the provided parameters"""
        fdat = self._feffdat
        if fdat.reff < 0.05:
            print('reff is too small to calculate chi(k)')
            return
        # make sure we have a k array
        if k is None:
            if kmax is None:
                kmax = 30.0
            kmax = min(max(fdat.k), kmax)
            if kstep is None: kstep = 0.05
            k = kstep * np.arange(int(1.01 + kmax/kstep), dtype='float64')
        if not self.use:
            self.k = k
            self.p = k
            self.chi = 0.0 * k
            self.chi_imag = 0.0 * k
            return
        reff = fdat.reff
        # get values for all the path parameters
        (degen, s02, e0, ei, deltar, sigma2, third, fourth)  = \
                self.__path_params(degen=degen, s02=s02, e0=e0, ei=ei,
                                 deltar=deltar, sigma2=sigma2,
                                 third=third, fourth=fourth)

        # create e0-shifted energy and k, careful to look for |e0| ~= 0.
        en = k*k - e0*ETOK
        if min(abs(en)) < SMALL_ENERGY:
            try:
                en[np.where(abs(en) < 1.5*SMALL_ENERGY)] = SMALL_ENERGY
            except ValueError:
                pass
        # q is the e0-shifted wavenumber
        q = np.sign(en)*np.sqrt(abs(en))

        # lookup Feff.dat values (pha, amp, rep, lam)
        if interp.startswith('lin'):
            pha = np.interp(q, fdat.k, fdat.pha)
            amp = np.interp(q, fdat.k, fdat.amp)
            rep = np.interp(q, fdat.k, fdat.rep)
            lam = np.interp(q, fdat.k, fdat.lam)
        else:
            pha = self.spline_coefs['pha'](q)
            amp = self.spline_coefs['amp'](q)
            rep = self.spline_coefs['rep'](q)
            lam = self.spline_coefs['lam'](q)

        if debug:
            self.debug_k   = q
            self.debug_pha = pha
            self.debug_amp = amp
            self.debug_rep = rep
            self.debug_lam = lam

        # p = complex wavenumber, and its square:
        pp   = (rep + 1j/lam)**2 + 1j * ei * ETOK
        p    = np.sqrt(pp)

        # the xafs equation:
        cchi = np.exp(-2*reff*p.imag - 2*pp*(sigma2 - pp*fourth/3) +
                      1j*(2*q*reff + pha +
                          2*p*(deltar - 2*sigma2/reff - 2*pp*third/3) ))

        cchi = degen * s02 * amp * cchi / (q*(reff + deltar)**2)
        cchi[0] = 2*cchi[1] - cchi[2]
        # outputs:
        self.k = k
        self.p = p
        self.chi = cchi.imag
        self.chi_imag = -cchi.real



def path2chi(path, paramgroup=None, **kws):
    """calculate chi(k) for a Feff Path,
    optionally setting path parameter values
    output chi array will be written to path group

    Parameters:
    ------------
      path:        a FeffPath Group
      params:      lmfit Parameters or larch ParameterGroup
      kmax:        maximum k value for chi calculation [20].
      kstep:       step in k value for chi calculation [0.05].
      k:           explicit array of k values to calculate chi.

    Returns:
    ---------
      None - outputs are written to path group

    """
    if not isNamedClass(path, FeffPathGroup):
        msg('%s is not a valid Feff Path' % path)
        return
    path.calc_chi_from_params(paramgroup, **kws)


def ff2chi(paths, group=None, paramgroup=None, k=None, kmax=None,
            kstep=0.05,  **kws):
    """sum chi(k) for a list of FeffPath Groups.

    Parameters:
    ------------
      paths:       a list of FeffPath Groups or dict of {label: FeffPathGroups}
      paramgroup:  a Parameter Group for calculating Path Parameters [None]
      kmax:        maximum k value for chi calculation [20].
      kstep:       step in k value for chi calculation [0.05].
      k:           explicit array of k values to calculate chi.
    Returns:
    ---------
       group contain arrays for k and chi

    This essentially calls path2chi() for each of the paths in the
    `paths` and writes the resulting arrays to group.k and group.chi.

    """
    if isinstance(paramgroup, Parameters):
        params = paramgroup
    else:
        params = group2params(paramgroup)


    if isinstance(paths, (list, tuple)):
        pathlist = paths
    elif isinstance(paths, dict):
        pathlist = list(paths.values())
    else:
        raise ValueErrror('paths must be list, tuple, or dict')

    if len(pathlist) == 0:
        return Group(k=np.linspace(0, 20, 401),
                     chi=np.zeros(401, dtype='float64'))

    for path in pathlist:
        if not isNamedClass(path, FeffPathGroup):
            print('%s is not a valid Feff Path' % path)
            return
        path.create_path_params(params=params)
        path._calc_chi(k=k, kstep=kstep, kmax=kmax)
    k = pathlist[0].k[:]
    out = np.zeros_like(k)
    for path in pathlist:
        out += path.chi

    if group is None:
        group = Group()
    group.k = k
    group.chi = out
    return group

def feffpath(filename='', label='', feffrun='', s02=None, degen=None,
             e0=None,ei=None, deltar=None, sigma2=None, third=None,
             fourth=None, use=True, **kws):
    """create a Feff Path Group from a *feffNNNN.dat* file.

    Parameters:
    -----------
      filename:  name (full path of) *feffNNNN.dat* file
      label:     label for path   [file name]
      degen:     path degeneracy, N [taken from file]
      s02:       S_0^2    value or parameter [1.0]
      e0:        E_0      value or parameter [0.0]
      deltar:    delta_R  value or parameter [0.0]
      sigma2:    sigma^2  value or parameter [0.0]
      third:     c_3      value or parameter [0.0]
      fourth:    c_4      value or parameter [0.0]
      ei:        E_i      value or parameter [0.0]
      feffrun:   label for Feff run          [parent folder of Feff.dat file]
      use :      use in sum of paths         [True]

    For all the options described as **value or parameter** either a
    numerical value or a Parameter (as created by param()) can be given.

    Returns:
    ---------
        a FeffPath Group.
    """
    return FeffPathGroup(filename=filename, label=label, feffrun=feffrun,
                         s02=s02, degen=degen, e0=e0, ei=ei, deltar=deltar,
                         sigma2=sigma2, third=third, fourth=fourth, use=use)

def use_feffpath(pathcache, label, degen=None, s02=None,  e0=None,ei=None,
                 deltar=None, sigma2=None, third=None, fourth=None, use=True):
    """use a copy of a Feff Path from a cache of feff paths - a simply dictionary
    keyed by the path label, and to support in-memory paths, not read from feff.dat files

    Parameters:
    -----------
      pathcache: dictionary of feff paths
      label:     label for path -- the dictionary key
      degen:     path degeneracy, N [taken from file]
      s02:       S_0^2    value or parameter [1.0]
      e0:        E_0      value or parameter [0.0]
      deltar:    delta_R  value or parameter [0.0]
      sigma2:    sigma^2  value or parameter [0.0]
      third:     c_3      value or parameter [0.0]
      fourth:    c_4      value or parameter [0.0]
      ei:        E_i      value or parameter [0.0]
      use:       whether to use path in sum [True]
      """
    path = deepcopy(pathcache[label])
    path.use = use
    path.init_path_params(s02=s02, degen=degen, e0=e0, ei=ei,
                         deltar=deltar, sigma2=sigma2, third=third,
                         fourth=fourth)
    return path
