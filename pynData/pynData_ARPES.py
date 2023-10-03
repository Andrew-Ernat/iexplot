#==============================================================================
#==============================================================================
# Domain specific applications - ARPES extensions for pynData
#      Data Format (data.shape = (z,y,x))
#          Axis x:  Energy (eV)
#          Axis y:  Angle (Degrees)
#          Axis z:  Scan axis (Theta, hv, Beta ...)
#
#       .KEscale:   original KE scale
#       .angScale:  original angle scale of detector
#       .angOffset: offset of orignal angular scaling, used for book keeping
#       .slitDir:   slit direction, 'H' or 'V'
#       .thetaX:    polar angle
#       .thetaY:    other angle
#       .hv:        photon energy in eV
#       .wk:        work function of analyzer in eV, setWk(d,val) to update-
#       .EDC        angle integrated pynData object
#       .MDC        energy integrated pynData object
#==============================================================================

#==============================================================================
# imports
#==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import ast
from scipy import interpolate


from iexplot.pynData.pynData import nData, nData_h5Group_r, nData_h5Group_w, nstack
from iexplot.utilities import *
from iexplot.plotting import plot_1D
#==============================================================================
# Global variables in science
#==============================================================================

kB = 8.6173423e-05          # Boltzmann k in eV/K
me = 5.68562958e-32         # Electron mass in eV*(Angstroms^(-2))*s^2
hbar = 6.58211814e-16       # hbar in eV*s
hc_over_e = 12.3984193      # hc/e in keV⋅A
hc_e = 1239.84193           # hc/e in eV*A

#==============================================================================
# functions to convert to and from k-space
#==============================================================================

def theta_to_kx(KE, thetaX):
    '''
    thetaX = polar angle
    kx = c*sqrt(KE)*sin(thetaX)
    '''

    c = np.sqrt(2*me)/hbar
    kx = c*np.sqrt(KE)*np.sin(thetaX*np.pi/180.)

    return kx


def theta_to_ky(KE, thetaX, thetaY):
    '''
    thetaX = polar angle
    thetaY = other angle
    ky = c*sqrt(KE)*cos(thetaX)*sin(thetaY)
    '''
    c = np.sqrt(2*me)/hbar
    ky = c*np.sqrt(KE)*np.cos(thetaX*np.pi/180.)*np.sin(thetaY*np.pi/180.)
    
    return ky


def theta_to_kz(KE, thetaX, thetaY, V0=10):
    '''
    thetaX = polar angle
    thetaY = other angle
    V0=inner potential
    kz = c*sqrt(V0-KE*(sin(thetaX)+cos(thetaX)*sin(thetaY)+1))
    '''
    c = np.sqrt(2*me)/hbar
    kz = c*np.sqrt(V0-KE*(np.sin(thetaX)+np.cos(thetaX)*np.cos(thetaY)+1)) #equation is probably wrong JM look into this

    return kz
    

def k_to_thetaX(KE, kx):
    '''
    
    '''
    
    c = np.sqrt(2*me)/hbar
    thetaX = np.arcsin(kx/(c*np.sqrt(KE)))
    
    return thetaX

def k_to_thetaY(KE, kx, ky):
    '''
    
    '''
    
    c = np.sqrt(2*me)/hbar
    thetaY = np.arcsin(ky/(c*np.sqrt(KE)*np.cos(np.arcsin(kx/(c*np.sqrt(KE))))))
    
    return thetaY
    


###############################################################################################
class nARPES(nData):
    def _nARPESattributes(self,metadata,**kwargs):
        """
        self is an pynData object and we will add the following attribute from the metadata dictionary
            'KEscale':[],     # original kinetic scale
            'BEscale':[],     # calculated binding energy scale from KEscale, hv and wk
            'angScale':[],    # original angle scale of detector
            'angOffset':0     # offset of orignal angular scaling, used for book keeping
            'slitDir':'V'     # slit direction, 'H' or 'V'
            'thetaX':0,       # polar angle
            'thetaY':0,       # other angle
            'hv':22.0,        # photon energy in eV
            'wk':4.0,         # work function of analyzer in eV, setWk(d,val) to update
            'EDC'/'MDC':      # pynData EDC/MDC data
            'spectraInfo':    # dictionary with detector info         
        """
        kwargs.setdefault("debug",False)
        self.KEscale=np.empty(1)
        self.BEscale=np.empty(1)
        self.angScale=np.empty(1)
        self.angOffset=0
        self.slitDir='V'
        self.thetaX=0
        self.thetaY=0
        self.hv=22.0
        self.wk=4.0
        self.EDC=nData(np.empty(1))
        self.MDC=nData(np.empty(1))
        self.spectraInfo={}
        
        if kwargs['debug']:
            print("\n _nARPESattributes")
            print(vars(self))
        for key in vars(self):
            if key in metadata:
                if metadata != None:
                    setattr(self, key, metadata[key])    

        self._BE_calc()

    def get_nARPESattributes(self):
        keys = ['KEscale','BEscale','angScale','angOffset','slitDir','thetaX','thetaY','hv','wk','EDC','MDC','spectraInfo']
        attr = {}
        for key in keys:
            attr[key] = getattr(self,key)
        return attr
 

    #==============================================================================
    # converting between KE and BE scaling
    #==============================================================================
    def scaleKE(self):
        """
        sets the scaling of the specificed axis to original KE scaling
        also sets the EDC scaling to the orginal KE scaling
        """
        self.updateAx('x',np.array(self.KEscale),"Kinetic Energy (eV)")
        self.EDC.updateAx('x',np.array(self.KEscale),"Kinetic Energy (eV)")
    
    def _BE_calc(self,wk=None, hv=None):
        """
        calculates the binding energy and updates self.BEscale
        where BE = hv - KE - wk (above EF BE is negative); KE is orginal KE scaling
            wk=None uses  .wk for the work function (wk ~ hv - KE_FermiLevel)
            hv=None uses  .hv for the photon energy
    
        """
        if wk is None:
            wk=self.wk
        if hv is None:
            hv=self.hv

        KE=np.array(self.KEscale)
        try:
            self.BEscale = hv-wk-KE
        except:
            print('BEscale not calculated')
            print(hv,wk)

    def scaleBE(self,wk=None, hv=None):
        """
        sets the scaling of the specificed axis to original BE scaling
        also sets the EDC scaling to the orginal BE scaling
        where BE = hv - KE - wk (above EF BE is negative); KE is orginal KE scaling
            wk=None uses  .wk for the work function (wk ~ hv - KE_FermiLevel)
            hv=None uses  .hv for the photon energy
    
        """
        self._BE_calc()
        BE = self.BEscale

        self.updateAx('x',BE,"Binding Energy (eV)")
        self.EDC.updateAx('x',BE,"Binding Energy (eV)")
    
    def set_wk(self,val):
        """
        updates the work function can be a single value or an array of the same length as KE
        recalucates the binding energy
        """
        self.wk(val)
        self._BE_calc()
    
#==============================================================================
# adjusting angle scaling
#==============================================================================
    
    def scaleAngle(self,delta=0):
        """
        changest the angle scaling of the data and the MDC
        based on the orginal angle scale and angOffset
            newScale = angScale + angOffset + delta;
            delta=(newCoor-oldCoor); can be value or an array of the same length as angScale
        
        """
        angScale=np.array(self.angScale)
        newScale= angScale + self.angOffset + delta
        self.updateAx('y',newScale,"Angle (deg)")
        self.MDC.updateAx('x',newScale,"Angle (deg)")
    
#==============================================================================
# calculating k scaling
#==============================================================================

    def kx_min_max(self, KE, thetaX):
        '''
        KE type: numpy array
        thetaX type: numpy array
        '''
        kx_max = theta_to_kx(np.max(KE),np.max(thetaX))
        kx_min = theta_to_kx(np.max(KE),np.min(thetaX))
        
        return kx_min, kx_max
        
        
    def ky_min_max(self, KE, thetaX, thetaY):
        '''
        KE type: numpy array
        thetaX type: numpy array
        thetaY type: numpy array
        '''
        ky_max = theta_to_ky(np.max(KE),np.min(thetaX),np.max(thetaY))
        ky_min = theta_to_ky(np.max(KE),np.max(thetaX),np.min(thetaY))
        
        return ky_min, ky_max

    def kScale(self, KE, thetaY, thetaX=np.empty):
        
        kx_min, kx_max = nARPES.kx_min_max(nARPES, KE, thetaX)
        kx_scale = np.linspace(kx_min, kx_max,len(thetaX))
        
        ky_min, ky_max = nARPES.ky_min_max(nARPES, KE, thetaX, thetaY)
        ky_scale = np.linspace(ky_min, ky_max,len(thetaY))
        
        return kx_scale, ky_scale
    
def _calc_BEscale_by_hand(d,KE,hv=None,wk=None):  
    """#when we fix types we can use _calc_BE  """
    if hv.all() == None:
        try:
            hv = d.hv
        except:
            print("You need to specify hv, can't read metadata")
            return
    if wk == None:
        try:
            wk = d.wk
        except:
            print("You need to specify wk, can't read metadata")
            return
    BE = hv-wk-KE 
    return BE
   
def kmap_scan_theta(d,hv=None,wk=None):
    '''
    d type: pynData stack (degrees, KE, theta)
    returns pynData (ky, kx, BE)
    '''
    KE = d.scale['x']
    thetaY = d.scale['y']
    thetaX = d.scale['z']
    
    org = d.data #data(thetaY, KE, thetaX)
    
    kx_scale, ky_scale = nARPES.kScale(nARPES, KE, thetaY, thetaX)
    
    new = np.zeros((len(ky_scale),len(KE),len(kx_scale)))
    new = interpolate.RegularGridInterpolator((ky_scale,KE,kx_scale),org, method = 'linear')
    
    dnew = nData(new.values.transpose(0,2,1))
    
    BE = _calc_BEscale_by_hand(d,KE,hv,wk)
    nData.updateAx(dnew,'x',kx_scale,'kx')
    nData.updateAx(dnew,'y',ky_scale,'ky')
    nData.updateAx(dnew,'z',BE,'BE')
    
    return dnew

def kmap_scan_hv(d,wk):
    '''
    d type: pynData stack (degrees, KE, hv)
    returns pynData (ky, BE, hv)
    '''
    KE = d.scale['x']
    thetaY = d.scale['y']
    hv = d.scale['z']
    thetaX = np.zeros(len(hv))

    org = d.data #data(thetaY, KE, hv)

    kx_scale, ky_scale = nARPES.kScale(nARPES, KE, thetaY, thetaX)

    new = np.zeros((len(ky_scale),len(KE),len(hv)))
    new = interpolate.RegularGridInterpolator((ky_scale,KE,hv),org, method = 'linear')

    dnew = nData(new.values.transpose(1,0,2))
    
    BE = _calc_BEscale_by_hand(d,KE,hv,4.6)
    nData.updateAx(dnew,'x',ky_scale,'ky')
    nData.updateAx(dnew,'y',BE,'BE')
    nData.updateAx(dnew,'z',hv,'hv')

    return dnew

###############################################################################################
def plotEDCs(*d,**kwargs):
    """
    Simple plot for EDCs 

    *d  set of pynData_ARPES data
    
    uses the current scale['x'] for the energy axis
        d.scaleKE() or d.scaleBE() to set

    kwargs:
        matplotlib.plot kwargs
    """
    for di in list(d):
        x = di.EDC.scale['x']
        y = di.EDC.data
        kwargs.update({'xlabel':di.EDC.unit['x']})
        plot_1D(x,y,**kwargs)
    return

def plotMDCs(*d,**kwargs):
    """
    Simple plot for EDCs 

    *d  set of pynData_ARPES data
    
    uses the current scale['x'] for the energy axis
        d.scaleKE() or d.scaleBE() to set

    kwargs:
        matplotlib.plot kwargs
    """
    for di in list(d):
        x = di.MDC.scale['x']
        y = di.MDC.data
        kwargs.update({'xlabel':di.MDC.unit['x']})
        plot_1D(x,y,**kwargs)
    return
        
##########################################
# generalized code for saving and loading as part of a large hd5f -JM 4/27/21
# creates/loads subgroups    
##########################################
def nARPES_h5Group_w(nd,parent,name):
    """
    for an nData object => nd
    creates an h5 group with name=name within the parent group:
        with all ndata_ARPES attributes saved                   
    """
    #saving ndata array
    g=nData_h5Group_w(nd,parent,name)
    
    #EDC/MDC
    nData_h5Group_w(nd.EDC,g,"EDC")
    nData_h5Group_w(nd.EDC,g,"MDC")
    
    for attr in ['hv','wk','thetaX','thetaY','KEscale','angScale','angOffset']:
        if type(getattr(nd,attr)) == type(None):
            g.create_dataset(attr, data=np.array([]) , dtype='f')
        else:
            g.create_dataset(attr, data=np.array(getattr(nd,attr)) , dtype='f')
    for attr in ['slitDir']:
        g.attrs[attr]=str(getattr(nd,attr))
    return g

def nARPES_h5Group_r(h):
    """           
    """
    d=nData_h5Group_r(h)
    
    #EDC/MDC
    d.EDC=nData_h5Group_r(h['EDC'])
    d.MDC=nData_h5Group_r(h['MDC']) 
    
    
    #val=ast.literal_eval(h5['mda']['mda_'+str(scanNum)]['header'][hkey].attrs[key])
    #for att in dir(your_object):
        #print (att, getattr(your_object,att))
    for attr in ['hv','wk','thetaX','thetaY','KEscale','angScale','angOffset']:
        setattr(d,attr,h[attr])
    for attr in ['slitDir','fpath']:
        if attr in h.attrs:
            setattr(d,attr,ast.literal_eval(h.attrs[attr]))
    return d   

