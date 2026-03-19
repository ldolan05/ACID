# This code existed just before "getting final profiles" in ACID.py ACID function

fig_opt = 'n'
if fig_opt =='y':

    # plots random models from flat_samples - lets you see if it's converging
    plt.figure()
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        mdl = model_func(sample, x)
        mdl1 = 0
        for i in np.arange(k_max, len(sample)-1):
            mdl1 = mdl1+sample[i]*((a*x)+b)**(i-k_max)
        mdl1 = mdl1*sample[-1]
        plt.plot(x, mdl1, "C1", alpha=0.1)
        plt.plot(x, mdl, "g", alpha=0.1)
    plt.scatter(x, y, color = 'k', marker = '.', label = 'data')
    plt.xlabel("wavelengths")
    plt.ylabel("flux")
    plt.title('mcmc models and data')
    plt.savefig('figures/mcmc_and_data.png')

    prof_flux = np.exp(profile)-1

    # plots the mcmc profile - will have extra panel if it's for data
    fig, ax0 = plt.subplots()
    ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
    zero_line = [0]*len(velocities)
    ax0.plot(velocities, zero_line)
    ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
    ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
    ax0.set_xlabel('velocities')
    ax0.set_ylabel('optical depth')
    secax = ax0.secondary_yaxis('right', functions = (utils.od2flux, utils.flux2od))
    secax.set_ylabel('flux')
    ax0.legend()
    plt.savefig('figures/profile_%s'%(run_name))

    # plots mcmc continuum fit on top of data
    plt.figure('continuum fit from mcmc')
    plt.plot(x, y, color = 'k', label = 'data')
    mdl1 =0
    for i in np.arange(0, len(poly_cos)-1):
        mdl1 = mdl1+poly_cos[i]*((a*x)+b)**(i)
    mdl1 = mdl1*poly_cos[-1]
    plt.plot(x, mdl1, label = 'mcmc continuum fit')
    mdl1_poserr =0
    for i in np.arange(0, len(poly_cos)-1):
        mdl1_poserr = mdl1_poserr+(poly_cos[i]+poly_cos_err[i])*((a*x)+b)**(i)
    mdl1_poserr = mdl1_poserr*poly_cos[-1]
    mdl1_neg =0
    for i in np.arange(0, len(poly_cos)-1):
        mdl1_neg = mdl1_neg+(poly_cos[i]-poly_cos_err[i])*((a*x)+b)**(i)
    mdl1_neg = mdl1_neg*poly_cos[-1]
    plt.fill_between(x, mdl1_neg, mdl1_poserr, alpha = 0.3)
    mdl1_err =abs(mdl1-mdl1_neg)
    plt.legend()
    plt.title('continuum from mcmc')
    plt.xlabel("wavelengths")
    plt.ylabel("flux")
    plt.savefig('figures/cont_%s'%(run_name))

    mcmc_inputs = np.concatenate((profile, poly_cos))
    mcmc_mdl = model_func(mcmc_inputs, x)

    residuals_2 = (y+1) - (mcmc_mdl+1)

    fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
    non_masked = tuple([yerr<10])
    #ax[0].plot(x, y+1, color = 'r', alpha = 0.3, label = 'data')
    #ax[0].plot(x[non_masked], mcmc_mdl[non_masked]+1, color = 'k', alpha = 0.3, label = 'mcmc spec')
    ax[1].scatter(x[non_masked], residuals_2[non_masked], marker = '.')
    ax[0].plot(x, y, 'r', alpha = 0.3, label = 'data')
    ax[0].plot(x, mcmc_mdl, 'k', alpha =0.3, label = 'mcmc spec')
    residual_masks = tuple([yerr>=100000000000000])

    #residual_masks = tuple([yerr>10])
    ax[0].scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    ax[0].legend(loc = 'lower right')
    #ax[0].set_ylim(0, 1)
    #plotdepths = -np.array(line_depths)
    #ax[0].vlines(line_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
    ax[1].plot(x, residuals_2, '.')
    #ax[1].scatter(x[residual_masks], residuals_2[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    z_line = [0]*len(x)
    ax[1].plot(x, z_line, '--')
    plt.savefig('figures/forward_%s'%(run_name))
    

    fig, ax0 = plt.subplots()
    ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
    zero_line = [0]*len(velocities)
    ax0.plot(velocities, zero_line)
    ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
    ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
    ax0.set_xlabel('velocities')
    ax0.set_ylabel('optical depth')
    ax0.legend()
    plt.savefig('figures/final_profile_%s'%(run_name))