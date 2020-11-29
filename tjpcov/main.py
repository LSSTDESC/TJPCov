import pyccl as ccl
import sacc
from tjpcov import wigner_transform, bin_cov, parse
import numpy as np
d2r = 180/np.pi

class CovarianceCalculator():
    def __init__(self,
                 cosmo_fn=None,
                 sacc_fn_xi=None,  # real space
                 sacc_fn_cl=None,  # harmonic space
                 window_fn=None):
        """
        Is reading all config from a single yaml a good option?
        - sacc passing values after scale cuts and no angbin_edges
        - angbin_edges necessary for bin averages
        - Find firecrown's way to handle survey features

        Parameters
        ----------
        cosmo_fn : None, str
            path to CCL yaml file 
            Fiducial Cosmology used in calculations
        sacc_fn_xi/cl : None, str
            path to sacc file yaml
            sacc object containing the tracers and binning to be used in covariance calculation
        window : None, dict, float
            If None, used window function specified in the sacc file
            if dict, the keys are sacc tracer names and values are either HealSparse inverse variance
            maps 
            if float it is assumed to be f_sky value
        """
        if isinstance(cosmo_fn, str):
            try:
                cosmo = ccl.Cosmology.read_yaml(cosmo_fn)
            except Exception as err:
                print(f"Error in ccl cosmology loading from yaml \n{err}")
        else:
            cosmo = cosmo_fn

        if isinstance(sacc_fn_cl, str):
            try:
                cl_data = sacc.Sacc.load_fits(sacc_fn_cl)
            except Exception as err:
                print(f"Error in sacc loading from yaml \n{err}")
        else:
            cl_data = sacc_fn_cl  # fao assert here

        if isinstance(sacc_fn_xi, str):
            try:
                xi_data = sacc.Sacc.load_fits(sacc_fn_xi)
            except Exception as err:
                print(f"Error in sacc loading from yaml \n{err}")
        else:
            xi_data = sacc_fn_xi  # fao assert here

        # fao Set this inside get_ell_theta ?
        ell, ell_bins, ell_edges = None, None, None
        # theta input in arcmin
        theta, theta_bins, theta_edges = None, None, None

        self.cosmo = cosmo
        self.cl_data = cl_data
        self.xi_data = xi_data
        self.window_fn = window_fn  # windown handler TBD

        # fix this for the general case:
        ell_list = self.get_ell_theta(cl_data,
                                      'galaxy_density_cl',
                                      ('lens0', 'lens0'),
                                      'linear')
        # fix this for the sacc file case:
        th_list = self.set_ell_theta(25, 250, 20, do_xi=True)

        self.theta,  self.theta_edges = th_list
        self.theta_bins = np.sqrt(self.theta_edges[1:]*self.theta_edges[:-1])
        self.ell, self.ell_bins, self.ell_edges = ell_list

        print("Preparing WT...")
        self.WT = self.wt_setup(self.ell, self.theta)
        print("Done!")
        return



    def print_setup(self):
        """
        Prints current setup
        """
        cosmo = self.cosmo
        ell  = self.ell


    def set_ell_theta(self, ang_min, ang_max, n_ang, 
                        ang_scale='linear', do_xi=False):
        """
        Utility for return custom theta/ell bins (outside sacc)

        Parameters:
        -----------
        ang_min (int, float):
            if do_xi, ang is assumed to be theta (arcmin)
            if do_xi == False,  ang is assumed to be ell 
        Returns:
        --------
            (theta, theta_edges ) (degrees):
        """
        # FIXME:
        # Learning if sacc is passing this
        if not do_xi:
            ang_delta = (ang_max-ang_min)//n_ang
            ang_edges = np.arange(ang_min, ang_max+1, ang_delta)
            ang = np.arange(ang_min, ang_max + ang_delta)

        if do_xi:
            th_min = ang_min/60  # in degrees
            th_max = ang_max/60
            n_th_bins = n_ang
            ang_edges = np.logspace(np.log10(th_min), np.log10(th_max),
                                    n_th_bins+1)
            th = np.logspace(np.log10(th_min*0.98), np.log10(1), n_th_bins*30)
            # binned covariance can be sensitive to the th values. Make sue
            # you check convergence for your application
            th2 = np.linspace(1, th_max*1.02, n_th_bins*30)

            ang = np.unique(np.sort(np.append(th, th2)))

        return ang, ang_edges



    def get_ell_theta(self, two_point_data, data_type, tracer_comb, ang_scale, 
                        do_xi=False):
        """
        Get ell or theta for bins given the sacc object 
        For now, presuming only log and linear bins 

        Parameters:
        -----------

        Returns:
        --------
        """
        ang_name = "ell" if not do_xi else 'theta'

        # assuming same ell for all bins:
        data_types = two_point_data.get_data_types()
        ang_bins = two_point_data.get_tag(ang_name, data_type=data_types[0],
                                          tracers=tracer_comb)

        ang_bins = np.array(ang_bins)

        angb_min, angb_max = ang_bins.min(), ang_bins.max()
        if ang_name == 'theta':
            # assuming log bins
            del_ang = (ang_bins[1:]/ang_bins[:-1]).mean()
            ang_scale = 'log'
            assert 1 == 2

        elif ang_name == 'ell':
            # assuming linear bins
            del_ang = (ang_bins[1:] - ang_bins[:-1])[0]
            ang_scale = 'linear'

        ang, ang_edges = self.set_ell_theta(angb_min-del_ang/2,
                                       angb_max+del_ang/2,
                                       len(ang_bins),
                                       ang_scale=ang_scale, do_xi=do_xi)
        # Sanity check
        if ang_scale == 'linear':
            assert np.allclose((ang_edges[1:]+ang_edges[:-1])/2, ang_bins), \
                "differences in produced ell/theta"
        return ang, ang_bins, ang_edges


    def wt_setup(self, ell, theta):
        """
        Set this up once before the covariance evaluations

        Parameters:
        -----------
        ell (array): array of multipoles
        theta ( array): array of theta in degrees

        Returns:
        --------
        """
        # ell = two_point_data.metadata['ell']
        # theta_rad = two_point_data.metadata['th']*d2r
        # get_ell_theta()


        WT_factors={}
        WT_factors['lens','source']=(0,2)
        WT_factors['source','lens']=(2,0) #same as (0,2)
        WT_factors['source','source']={'plus':(2,2),'minus':(2,-2)}
        WT_factors['lens','lens']=(0,0)

        ell = np.array(ell)
        if not np.alltrue(ell>1):
            # fao check warnings in WT for ell < 2
            print("Removing ell=1 for Wigner Transformation")
            ell = ell[(ell>1)]

        WT_kwargs={'l': ell,
                   'theta': theta*d2r,
                   's1_s2':[(2,2),(2,-2),(0,2),(2,0),(0,0)]}


        WT=wigner_transform(**WT_kwargs)
        return WT



    def get_cov_WT_spin(tracer_comb=None):
        """
        Parameters:
        -----------
        tracer_comb (str, str): tracer combination in sacc format

        Returns:
        --------

        """
    #     tracers=tuple(i.split('_')[0] for i in tracer_comb)
        tracers=[]
        for i in tracer_comb:
            if 'lens' in i:
                tracers+=['lens']
            if 'src' in i:
                tracers+=['source']
        return WT_factors[tuple(tracers)]



    def get_tracer_info(self,two_point_data={}):
            """
            Creates CCL tracer objects and computes the noise for all the tracers
            Check usage: Can we call all the tracer at once?

            Parameters:
            -----------
                two_point_data (sacc obj):

            Returns:
            --------
                ccl_tracers: ccl obj
                    ccl.WeakLensingTracer or ccl.NumberCountsTracer
                tracer_Noise ({dict: float}): 
                    shot (shape) noise for lens (sources)
            """
            ccl_tracers = {}
            tracer_Noise = {}
            for tracer in two_point_data.tracers:
                tracer_dat = two_point_data.get_tracer(tracer)
                z = tracer_dat.z

                # FIXME: Following should be read from sacc dataset.--------------
                Ngal = 26.  # arc_min^2
                sigma_e = .26
                b = 1.5*np.ones(len(z))  # Galaxy bias (constant with scale and z)
                AI = .5*np.ones(len(z))  # Galaxy bias (constant with scale and z)
                Ngal = Ngal*3600/d2r**2
                # ---------------------------------------------------------------

                dNdz = tracer_dat.nz
                dNdz /= (dNdz*np.gradient(z)).sum()
                dNdz *= Ngal

                if 'source' in tracer or 'src' in tracer:
                    ccl_tracers[tracer] = ccl.WeakLensingTracer(
                        self.cosmo, dndz=(z, dNdz), ia_bias=(z, AI))
                    # CCL automatically normalizes dNdz
                    tracer_Noise[tracer] = sigma_e**2/Ngal
                elif 'lens' in tracer:
                    tracer_Noise[tracer] = 1./Ngal
                    ccl_tracers[tracer] = ccl.NumberCountsTracer(
                        self.cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, b))
            return ccl_tracers, tracer_Noise


    def cl_gaussian_cov(self, tracer_comb1=None, tracer_comb2=None, ccl_tracers=None, tracer_Noise=None,
                        two_point_data=None, do_xi=False, xi_plus_minus1='plus', xi_plus_minus2='plus'): 
        """
        Compute a single covariance matrix for a given pair of C_ell or xi
        
        Returns:
        --------
            final:  covariance for C_ell
            final_b : covariance for xi
        """
        #fsky should be read from the sacc
        #tracers 1,2,3,4=tracer_comb1[0],tracer_comb1[1],tracer_comb2[0],tracer_comb2[1]
        #ell=two_point_data.metadata['ell']
        cosmo = self.cosmo

        if not do_xi:
            ell = two_point_data.get_cl
        cl={}
        cl[13] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[0]], ell)
        cl[24] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[1]], ell)
        cl[14] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[0]], ccl_tracers[tracer_comb2[1]], ell)
        cl[23] = ccl.angular_cl(cosmo, ccl_tracers[tracer_comb1[1]], ccl_tracers[tracer_comb2[0]], ell)
        
        SN={}
        SN[13]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[0]  else 0
        SN[24]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[1]  else 0
        SN[14]=tracer_Noise[tracer_comb1[0]] if tracer_comb1[0]==tracer_comb2[1]  else 0
        SN[23]=tracer_Noise[tracer_comb1[1]] if tracer_comb1[1]==tracer_comb2[0]  else 0
        
        if do_xi:
            norm=np.pi*4*two_point_data.metadata['fsky']
        else: #do c_ell
            norm=(2*ell+1)*np.gradient(ell)*two_point_data.metadata['fsky']

        coupling_mat={}
        coupling_mat[1324]=np.eye(len(ell)) #placeholder
        coupling_mat[1423]=np.eye(len(ell)) #placeholder
        
        cov={}
        cov[1324]=np.outer(cl[13]+SN[13],cl[24]+SN[24])*coupling_mat[1324]
        cov[1423]=np.outer(cl[14]+SN[14],cl[23]+SN[23])*coupling_mat[1423]
        
        cov['final']=cov[1423]+cov[1324]
        
        if do_xi:
            s1_s2_1=get_cov_WT_spin(tracer_comb=tracer_comb1)
            s1_s2_2=get_cov_WT_spin(tracer_comb=tracer_comb2)
            if isinstance(s1_s2_1,dict):
                s1_s2_1=s1_s2_1[xi_plus_minus1] 
            if isinstance(s1_s2_2,dict):
                s1_s2_2=s1_s2_2[xi_plus_minus2] 
            th,cov['final']=WT.projected_covariance2(l_cl=ell,s1_s2=s1_s2_1, s1_s2_cross=s1_s2_2,
                                                          cl_cov=cov['final'])

        cov['final']/=norm
        
        if do_xi:
            thb,cov['final_b']=bin_cov(r=th/d2r,r_bins=two_point_data.metadata['th_bins'],cov=cov['final']) 
        else:
            if two_point_data.metadata['ell_bins'] is not None:
                lb,cov['final_b']=bin_cov(r=ell,r_bins=two_point_data.metadata['ell_bins'],cov=cov['final']) 
                
    #     cov[1324]=None #if want to save memory
    #     cov[1423]=None #if want to save memory
        return cov



    #compute all the covariances and then combine them into one single giant matrix
    def get_all_cov(two_point_data={},do_xi=False):
        """
        Compute all the covariances and then combine them into one single giant matrix
        Parameters:
        -----------
        two_point_data (sacc obj): sacc object containg two_point data
        
        Returns:
        --------
        cov_full (Npt x Npt numpy array):
            Covariance matrix for all combinations. 
            Npt = (number of bins ) * (number of combinations)
        
        """
        #FIXME: Only input needed should be two_point_data, 
        #which is the sacc data file. Other parameters should be 
        #included within sacc and read from there."""
        
        ccl_tracers, tracer_Noise = get_tracer_info(two_point_data=two_point_data)
        
        tracer_combs=two_point_data.get_tracer_combinations() # we will loop over all these
        N2pt=len(tracer_combs)
        
        if two_point_data.metadata['ell_bins'] is not None:
            Nell_bins=len(two_point_data.metadata['ell_bins'])-1
        else:
            Nell_bins=len(two_point_data.metadata['ell'])
            
        if do_xi:
            Nell_bins=len(two_point_data.metadata['th_bins'])-1
        
        cov_full=np.zeros((Nell_bins*N2pt,Nell_bins*N2pt))
        
        #Fix this loop for uneven scale cuts (different N_ell)
        for i in np.arange(N2pt):
            print("{}/{}".format(i+1, N2pt))
            tracer_comb1=tracer_combs[i]
            indx_i=i*Nell_bins
            for j in np.arange(i,N2pt):
                tracer_comb2=tracer_combs[j]
                indx_j=j*Nell_bins
                cov_ij=cl_gaussian_cov(tracer_comb1=tracer_comb1,
                                       tracer_comb2=tracer_comb2,                            
                                       ccl_tracers=ccl_tracers,
                                       tracer_Noise=tracer_Noise,
                                       do_xi=do_xi,
                                       two_point_data=two_point_data)
                
                if do_xi or two_point_data.metadata['ell_bins'] is not None:
                    cov_ij=cov_ij['final_b']
                else:
                    cov_ij=cov_ij['final']
                    
                cov_full[indx_i:indx_i+Nell_bins, indx_j:indx_j+Nell_bins]=cov_ij
                cov_full[indx_j:indx_j+Nell_bins, indx_i:indx_i+Nell_bins]=cov_ij.T
        return cov_full


    def create_sacc_cov(output=None):
        """ Write created cov to a new sacc object
        """
        pass    


if __name__=="__main__":
    cosmo_filename = "../test/cosmo_desy1.yaml"
    # sxi = sacc.Sacc.load_fits("../examples/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits") 
    xi_fn = "../examples/des_y1_3x2pt/generic_xi_des_y1_3x2pt_sacc_data.fits"
    cl_fn = "../examples/des_y1_3x2pt/generic_cl_des_y1_3x2pt_sacc_data.fits"

    tjp = CovarianceCalculator(cosmo_fn=cosmo_filename, sacc_fn_cl=cl_fn, sacc_fn_xi=xi_fn)