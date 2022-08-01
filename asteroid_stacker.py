from pixell import enmap,utils, reproject, enplot, bunch
import numpy as np
import matplotlib.pyplot as plt
import os,sys,argparse 
from scipy.interpolate import interp1d
import math
import pandas as pd
import pickle as pk
import h5py
import time
from pathlib import Path

from astropy import wcs
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u
from astropy.io import fits
from astropy.table import QTable
import astropy.table as tb 
from astropy.time import Time 
from astroquery.jplhorizons import Horizons

import re
from numba import jit

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--index", type=int, default=-1)
parser.add_argument("-N", "--name",  type=str,   default=None)

args = parser.parse_args()

def get_desig(id_num):
    home = str(Path.home())
    with open(home+'/dev/minorplanets/asteroids.pk', 'rb') as f:
        df = pk.load(f)
        name = df['name'][id_num]
        desig = df['designation'][id_num]
    return desig, name

def tnoStamp(ra, dec, imap, width = 0.5):
    #Takes an individual stamp of a map at the requested ra/dec. We form the ratio later  
    #frhs is a matched filter map and kmap is the inverse variance per pixel
    #both maps are at a ~3 day cadence

    #Inputs: ra, dec in degrees, j2000
    #kmap or frhs, described above. They must have the same wcs
    #but if you're using this code and sigurd's maps they will
    #width, the desired width of the stamp in degrees

    #Output: a stamp, centered on ra/dec, with width width, where
    #each pixel in the map records the S/N. This stamp is for one
    #object, one 3 day map. These must then be stacked for each object
    #and then the objects can be stacked together

    #Find the pixel 
    coords = np.deg2rad(np.array((dec,ra)))
    ypix,xpix = enmap.sky2pix(imap.shape,imap.wcs,coords)
        
    #nans are formed when try to form S/n for pixels with no hits
    #I just set the S/N to 0 which I think is safe enough
    imap[~np.isfinite(imap)] = 0

    #Reproject will attempt to take a stamp at the desired location: if it can't
    #for whatever reason just return None. We don't want it throwing errors
    #while on the wall, however the skips should probably be checked after
    #try:
    stamp = reproject.thumbnails(imap, [coords])
    #except:
    #    return None
    
    return stamp

class OrbitInterpolator:
    '''
    Constructs a class that can predict, using an interpolation scheme, the location of an object given an identifier and time
    '''
    def __init__(self, table):
        '''
        Requires a table generated from astroquery, querying JPL HORIZONS. The 
        interpolation is done automatically, but, of course, only works
        if the table is sampled densely enough and in the time range for which
        the positions were queried
        '''
        self.table = table
        self.targets = np.unique(table['targetname'])

        self._construct_dictionary()

    def _interpolate_radec(self, target):
        '''
        "Hidden" function that constructs the interpolations for each target
        '''        
        table = self.table[self.table['targetname'] == target]
        zero = np.min(table['datetime_jd'])
        ra_interp = interp1d(table['datetime_jd'] - zero, table['RA'])
        dec_interp = interp1d(table['datetime_jd'] - zero, table['DEC'])
        delta_interp = interp1d(table['datetime_jd'] - zero, table['delta'])
        r_interp = interp1d(table['datetime_jd'] - zero, table['r'])

        return zero, ra_interp, dec_interp, delta_interp, r_interp

    def _construct_dictionary(self):
        '''
        "Hidden" function that creates the look-up dictionary of targets for simplicity of usage
        '''
        self.obj_dic = {}
            
        for j,i in enumerate(self.targets):
            z, ra, dec, delta, r = self._interpolate_radec(i)
            self.obj_dic[i] = {}
            self.obj_dic[i]['zero'] = z
            self.obj_dic[i]['RA'] = ra
            self.obj_dic[i]['DEC'] = dec
            self.obj_dic[i]['delta'] = delta
            self.obj_dic[i]['r'] = r

    def get_radec_dist(self, target, time):
        '''
        Specifying a target name (see self.obj_dic.keys() for a list of targets) and a time (in JD), finds
        the interpolated RA and Dec for the objects
        '''
        time = time + 2400000.5

        t_intep = time - self.obj_dic[target]['zero']
        
        ra = self.obj_dic[target]['RA'](t_intep)
        dec = self.obj_dic[target]['DEC'](t_intep)
        dist = self.obj_dic[target]['delta'](t_intep)
        r = self.obj_dic[target]['r'](t_intep)

        return ra, dec, dist, r

class QueryHorizons:
    '''
    Constructs a class that can query a bunch of positions for objects from JPL-Horizons given a set of times and an observatory location
    '''

    def __init__(self, time_start, time_end, observer_location, step = '1d'):
        '''
        Initialization function

        Arguments:
        - time_start: start time for the query, should be in MJD
        - time_end: end time for the query, should be in MJD
        - observer_location: location of the observer
        - step: time step for the ephemeris
        The simples way to get the observer location variable is via the
        list of IAU observatory codes: https://en.wikipedia.org/wiki/List_of_observatory_codes
        
        Custom locations are also accepted by JPL
        '''
        if type(time_start) == str: 
            t_st = Time(time_start, format='isot', scale='utc')
            
        else: 
            t_st = Time(time_start, format='mjd')
        self.time_start = t_st.utc.iso  
        if type(time_end) == str: 
            t_en = Time(time_end, format='isot', scale='utc')
            
        else: 
            t_en = Time(time_end, format='mjd')
            
        self.time_end = t_en.utc.iso 
        
        self.observer = observer_location

        self.step = step 


    def queryObjects(self, objects):
        '''
        Returns a table (and saves it on the object as well) for the provided list of objects
        '''
        self.table = [] 

        for i in objects:
            query = Horizons(id = i, location = self.observer, 
                epochs = {'start' : self.time_start, 'stop' : self.time_end, 'step' : self.step})

            eph = query.ephemerides()

            self.table.append(eph['RA', 'DEC', 'datetime_jd', 'targetname', 'delta', 'r'])
        
        self.table = tb.vstack(self.table)

        return self.table

class minorplanet():
    '''
    Constructs a class that includes everything we want to do with a minor planet
    '''
    def __init__(self, name, path = '/scratch/r/rbond/jorlo/actxminorplanets/sigurd/asteroids/', obs = 'W99', 
                 t_start ='2010-01-01T00:00:00', t_end='2020-01-01T00:00:00'):
        '''
        Requires a name for the object, as well as an observatory code which can be found at
        https://en.wikipedia.org/wiki/List_of_observatory_codes
        See the QueryHorizons class for full details of observatory code options
        '''
        self.name = name
        self.path = path
        self.obs = obs
        
        
        #Initialize orbit and flux stuff
        self.t_start = t_start
        self.t_end = t_end
        
        self.map_dict = {'pa4':{'150':0, '220':0},
                         'pa5':{'090':0, '150':0},
                         'pa6':{'090':0, '150':0}
                        }
        self.flux_dict = {'pa4':{'150':{'flux':0, 'var':0,}, '220':{'flux':0, 'var':0}},
                         'pa5':{'090':{'flux':0, 'var':0}, '150':{'flux':0, 'var':0}},
                         'pa6':{'090':{'flux':0, 'var':0}, '150':{'flux':0, 'var':0}}
                        }
        
        #We interpolate the orbit for the minor planet at initialization as it's fairly fast
        #Stacking is a method as it is much slower so we want to call it only when we actually care
        #self.eph_table = QueryHorizons(time_start=self.t_start, time_end=self.t_end, 
        #                               observer_location=obs, step = '1d').queryObjects([str(self.name)])
        
        #self.interp_orbit = OrbitInterpolator(self.eph_table)
        
        info = np.load('/gpfs/fs0/project/r/rbond/sigurdkn/actpol/ephemerides/objects/{}.npy'.format(self.name.capitalize())).view(np.recarray)
        self.orbit   = interp1d(info.ctime, [utils.unwind(info.ra*utils.degree), 
                                                    info.dec*utils.degree, info.r, info.rsun, info.ang*utils.arcsec], kind=3)

        
    def make_stack(self, pa, freq = 'f150', pol='T', weight_type=None):
        pol_dict = {'T':0, 'Q':1, 'U':2}
        pol = pol_dict[pol]

        if not os.path.isdir(self.path+'/'+self.name+'/'):
            print('Depth 1 stamps not made, please make them')
            return
        rho_stack = 0
        kappa_stack = 0
        
        rho_un = 0 
        kappa_un = 0
        
        rho_weight = 0
        kappa_weight = 0
        
        for i, dirname in enumerate(os.listdir(path=self.path+'/'+self.name+'/')):
            
            if dirname.find('kappa.fits') == -1: continue #Just look at the kappa maps
            if dirname.find(pa) == -1: continue
            if dirname.find(freq) == -1: continue #Check each file to see if it's the specified freq
            
           
          
            rhofile    = utils.replace(dirname, "kappa.fits", "rho.fits")
            infofile = utils.replace(dirname, "kappa.fits", "info.hdf")

            kappa = enmap.read_map(self.path+'/'+self.name+'/' + dirname)
            rho = enmap.read_map(self.path+'/'+self.name+'/' + rhofile)
            info = bunch.read(self.path+'/'+self.name+'/' +infofile)
            ctime0   = np.mean(info.period)


            ignore_ra, ignore_dec, delta_earth, delta_sun, ignore_ang = self.orbit(ctime0)
              
            
            if weight_type == 'flux':
                weight = 1/(delta_earth**2*delta_sun**2)
                
            else:
                weight = 1
                
            rho_stack += rho[pol,:,:]*weight
            kappa_stack += kappa[pol,:,:]*weight**2
            
            rho_un += rho[pol,:,:]
            kappa_un+= kappa[pol,:,:]
            
            rho_weight += weight
            kappa_weight += weight**2
             
        stack = (rho_stack/rho_weight) / (kappa_stack/kappa_weight)
        plt.imshow(stack)
        plt.colorbar()
        plt.title('Flux of {} from PA {} at Freq {}'.format(self.name, pa, freq))
        plt.savefig('/scratch/r/rbond/jorlo/actxminorplanets/sigurd/plots/stacks/{}_pa{}_{}.pdf'.format(self.name, pa, freq))
        
        self.map_dict[pa][freq] = stack
        
        self.flux_dict[pa][freq]['flux'] = stack.at([0,0])
        self.flux_dict[pa][freq]['var'] = np.mean((kappa_stack/kappa_weight)**0.5)
        

    def make_all_stacks(self, weight_type = 1, pol='T', directory = '/scratch/r/rbond/jorlo/actxminorplanets/sigurd/'):
        for pa_key in self.map_dict.keys():
            for freq_key in self.map_dict[pa_key].keys():
                self.make_stack(pa = pa_key, freq = freq_key, pol=pol, weight_type = weight_type)
                
        with open(directory+'fluxes/{}_flux_dict.pk'.format(self.name), 'wb') as f: 
            pk.dump(self.flux_dict, f)
        
           
    def plot_stack(self, directory = None, scale = None):
        plt.imshow(self.flux_stack)
        plt.xlabel('ra')
        plt.ylabel('dec')
        
        plt.title('Plot of {} Stack'.format(self.eph_table['targetname'][0]))
        if directory is not None:
            plt.savefig(directory + '{}_stamp.pdf'.format(str(self.eph_table['targetname'][0]).replace(' ', '_').replace('(','').replace(')','')))



if args.index == -1:
        name     = args.name 

else:
	desig, name = get_desig(args.index)


ast = minorplanet(name)
ast.make_all_stacks(weight_type='flux')





