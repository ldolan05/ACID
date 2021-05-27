import numpy as np
from scipy import linalg


def LSD(wavelengths, flux_obs, rms, linelist):
    
    vmax=25
    #deltav=1.1
    vmin=-vmax
    
    resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    deltav = resol1/(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
    #print(resol1)
    
    velocities=np.arange(vmin,vmax,deltav)
 
    id_matrix=np.identity(len(flux_obs))
    S_matrix=(1/rms)*id_matrix
    #print('Matrix S has been set up')
    
    #### This is the EXPECTED linelist (for a slow rotator of the same spectral type) ####
    linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
    wavelengths_expected1 =np.array(linelist_expected[:,0])
    depths_expected1 = np.array(linelist_expected[:,1])

    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)
    
    wavelengths_expected=[]
    depths_expected=[]
    for some in range(0, len(wavelengths_expected1)):
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max:
            wavelengths_expected.append(wavelengths_expected1[some])
            depths_expected.append(depths_expected1[some])
        else:
            pass
        
    #print('number of lines: %s'%len(depths_expected))
    #print('Expected linelist has been read in')
  
    blankwaves=wavelengths
    R_matrix=flux_obs
    #print('Matrix R has been set up')
    
    #delta_x=np.zeros([len(wavelengths_expected), (len(blankwaves)*len(velocities))])
    #print('Delta x')
    #print(np.shape(delta_x))
    
    alpha=np.zeros((len(blankwaves), len(velocities)))
    
    limit=np.max(velocities)*np.max(wavelengths_expected)/2.99792458e5
    
    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):
    
            diff=blankwaves[j]-wavelengths_expected[i]
            if abs(diff)<=(limit):
                vel=2.99792458e5*diff/wavelengths_expected[i]
                for k in range(0, len(velocities)):
                    x=(velocities[k]-vel)/deltav
                    if -1.<x and x<0.:
                        delta_x=(1+x)
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x 
                    elif 0.<=x and x<1.:
                        delta_x=(1-x)
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x 
            else:
                pass
    
    #print('Delta_x has been calculated')

    #print('Calculating Alpha...')
    
    #alpha=np.dot(depths_expected,delta_x)
    #alpha=np.reshape(alpha, (len(blankwaves), len(velocities)))
    
    #print('Alpha Calculated')
    
    S_squared=np.dot(S_matrix, S_matrix)
    alpha_transpose=(np.transpose(alpha))
    
    #print('Beginning deconvolution')
    RHS_1=np.dot(alpha_transpose, S_squared)
    RHS_final=np.dot(RHS_1, R_matrix )
    
    #print('RHS ready')
    
    LHS_preinvert=np.dot(RHS_1, alpha)
    LHS_prep=np.matrix(LHS_preinvert)
    
    #print('Beginning inversion')
    P,L,U=linalg.lu(LHS_prep)
    
    n=len(LHS_prep)
    B=np.identity(n)
    Z = linalg.solve_triangular(L, B, lower=True)
    X = linalg.solve_triangular(U, Z, lower=False)
    LHS_final = np.matmul(X,np.transpose(P))
    
    #print('Inversion complete')
    
    profile=np.dot(LHS_final, RHS_final)
    profile_errors_squared=np.diagonal(LHS_final)
    profile_errors=np.sqrt(profile_errors_squared)
    '''
    upper_errors = profile+profile_errors
    lower_errors = profile-profile_errors
    
    
    fig3 = plt.figure(3)
    plt.plot(velocities, profile, color = 'b')
    plt.fill_between(velocities, lower_errors, upper_errors, alpha=0.4)
    plt.xlabel('Velocity(km/s)')
    plt.ylabel('Flux(Arbitrary Units)')
    #fig3.savefig('%s.png'%spectrum)
 
    #stop = timeit.default_timer() #end point of the timer
    #print('Time: ', stop - start)
    
    plt.show()
    '''  
    return velocities, profile, profile_errors
    
############################################################################################################
