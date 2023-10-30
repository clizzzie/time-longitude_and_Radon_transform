import numpy as np
import matplotlib.pyplot as plt
import skimage
from cmcrameri import cm
from sklearn import linear_model


# Personal modules:
import array_to_dictionary_mod
import time_longitude_ledt002


date                = "19_October_2023"

# model               = "GGFSS70"
# model               = 'LSMOD.2'
# model               = "GGF100k"
# model               = "GaussianNoise"
# model               = "GGFMB"

model               = "ledt002"
# model               = "ledt039"
# model               = "Monika_Pm10_Ra350"

# model               = 'CALS10k.2'
# model               = "pfm9k.1a"
# model               = "model_test"


# Time longitude
r_surface           = 6371e3       # radius of Earth's surface in meters
r_CMB               = 3485e3       # radius of the CMB in meters
latitude            = 60
longitude_start     = -180
longitude_end       = 180
longitude_spacing   = 2
axisymm_state       = "axi_remove_TA"         # "axi_keep" "axi_remove_TA" "axi_remove_lin_reg"
high_pass           = "HP_off"
milliTesla          = "mT_off"

if model == "GGFSS70":              # Gauss coefficient degree and order
    degree          = 6
elif model == "GGFMB":
    degree          = 6
elif model == ("ledt002" or "ledt039"):
    degree          = 10
else:
    degree          = 10


# All Gauss coefficient time series should be processed the same
freq_cut                = 0
if model == "pfm9k.1a":
    HP_order            = 15*2
else:
    HP_order            = 20*2

TL_folder_mac           = '/Users/nclizzie/Documents/Research/Time_Longitude/'
TL_folder_Etote         = '/Volumes/clizzie_Etote/Time_longitude/'
Meeting_notes_folder    = '/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'

def main():
    """ Handles all the Radon drift analysis of time-longitude plots
    """ 

    if      model   == 'pfm9k.1a':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/PFM9k_ready_to_load/pfm9k1a_coeffs_surface_10years.txt')
    elif    model   == 'GaussianNoise':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/Gaussian_noise/Gaussian_noise.txt')
    elif    model   == 'CALS10k.2':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/CALS10k2_ready_to_load/CALS10k2_coeffs_surface_10years.txt')
    elif    model   == 'GGF100k':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGF100K_ready_to_load/ggf100k_coeffs_surface_10years.txt')
    elif    model   == 'GGFSS70':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGFSS70k_ready_to_load/GGFSS70k_coeffs_surface_10years.txt')
    elif    model   == 'LSMOD.2':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/LSMOD.2_ready_to_load/LSMOD.2_coeffs_surface_10years.txt')
    elif    model   == 'ledt002':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/NDS_ready_to_load/ledt2,39/NDS_coeffs_interpolated_ledt002.txt')
    elif    model   == 'ledt039':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/NDS_ready_to_load/ledt2,39/NDS_coeffs_interpolated_ledt039.txt')
    elif    model   == 'Monika_Pm10_Ra350':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/NDS_ready_to_load/monica/NDS_coeffs_surface_interpolated_Pm10_Ra350.txt')
    elif    model   == 'GGFMB':
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGFMB_ready_to_load/GGFMB_coeffs_surface_200years.txt')
        GC_Dict["year"] = np.rint(-GC_Dict["year"]*1000+2000).astype(int)
    else:
        print('sayonara')
        exit()


    if model == "ledt002":
        GC_Dict["ka"] = GC_Dict['year']
    else:
        GC_Dict["ka"] = np.abs(GC_Dict['year']-2000)/1000



    # # # # ------------------ Create initial time-longitude plot ------------------ #
    # year_start, year_end =  -7000, 2000
    # year_start, year_end =  -100000, 2000
    # year_start, year_end =  -48000, -28000      
    year_start, year_end =  0.75, 1.05        #ledt002
    # year_start, year_end =  0.855, 0.92     # 0.855, 0.92         # NatGeo
    # year_start, year_end =  -808000, -758000        # -810000, -760000 GGFMB

    year_start_index        = np.abs(GC_Dict['year']-year_start).argmin()
    year_end_index          = np.abs(GC_Dict['year']-year_end).argmin()

    print(axisymm_state)

    timelongitude_complete, years_complete, x_pixels, y_pixels = time_longitude.main(model, axisymm_state, high_pass, year_start, year_end, latitude, freq_cut , date)
    print('The first index of years: ', years_complete[0], '\n', 'The last index of years: ', years_complete[-1])
    

    # # # ------------------ Load time-longitude document ------------------ #

    # timelongitude_complete, years_complete, x_pixels, y_pixels  = load_TL_doc(TL_folder=TL_folder_Etote)
    timelongitude_complete, years_complete                      = check_TL_order(timelongitude_complete, years_complete)
    if high_pass == "HP_on":
        timelongitude, years, x_pixel, y_pixel                  = cut_off_edges_HP_filter(timelongitude_complete, years_complete)
    else:
        timelongitude, years, x_pixel, y_pixel                  = timelongitude_complete, years_complete, x_pixels, y_pixels
    if milliTesla == "mT_on":
        timelongitude                                           = convert_nT_to_mT(timelongitude)
    dt                                                          = np.abs(years_complete[0]-years_complete[1])

    print("The time interval is: ", dt)



    # # ------------------ Drift to projection angle ------------------ #
    # Drift to projection both negative and positive
    if   model == "GGF100k":
        drift_low, drift_high, drift_inc           = 0.01, 0.3, 0.005
    elif   model == "GaussianNoise":
        drift_low, drift_high, drift_inc           = 0.01, 0.3, 0.005
    elif model == ("LSMOD.2"):
        drift_low, drift_high, drift_inc           = 0.01, 0.3, 0.005
    elif model == ("GGFSS70"):
        drift_low, drift_high, drift_inc           = 0.01, 0.3, 0.005
    elif model == "ledt002":
        drift_low, drift_high, drift_inc           = 500, 40000 , 500
    elif model == 'pfm9k.1a':
        drift_low, drift_high, drift_inc           = 0.01, 0.5, 0.005
    else:
        print("Need to add drift rate")
        exit()
    drift               = np.concatenate( (np.arange(-drift_high, -drift_low+drift_inc, drift_inc), \
        np.arange(drift_low, drift_high+drift_inc, drift_inc)) )
    projection_degrees  = drift_to_projection(drift_low, drift_high, drift_inc, x_pixel, dt)

    print("The size of the drift is:", projection_degrees.shape)


    # # ------------------ Drift Determination for Moving Window ------------------ # recently for ledt002

    year_increments, year_window = 0.005, 0.05
    # year_increments, year_window = 200, 10000

    moving_window, mov_wind_years_cent, max_signal = moving_window_RT(timelongitude, years, year_increments, year_window, projection_degrees, dt, drift, normalization= 'no normalization')
    # np.savetxt(TL_folder_Etote+model+'/'+model+'_MA_HP'+str(freq_cut)+'.txt', moving_window)
    # np.savetxt(TL_folder_Etote+model+'/'+model+'_MA_wind_years_cent_HP'+str(freq_cut)+'.txt', mov_wind_years_cent)
    # np.savetxt(TL_folder_Etote+model+'/'+model+'_MA_max_signal_HP'+str(freq_cut)+'.txt', max_signal)


    # moving_window = np.loadtxt(TL_folder_Etote+model+'/'+model+'_MA_HP'+str(freq_cut)+'.txt')
    # mov_wind_years_cent = np.loadtxt(TL_folder_Etote+model+'/'+model+'_MA_wind_years_cent_HP'+str(freq_cut)+'.txt')
    # max_signal = np.loadtxt(TL_folder_Etote+model+'/'+model+'_MA_max_signal_HP'+str(freq_cut)+'.txt')

    plot_moving_window_and_max_signal_and_tilt_and_moment(GC_Dict, moving_window, max_signal, years, drift, mov_wind_years_cent, year_increments, year_window, dt)
    plt.savefig(Meeting_notes_folder+model+'_moving_window_'+axisymm_state+'_HP'+str(freq_cut)+'.pdf', format="pdf", dpi = 150)


    # # ------------------ Reverse flux and TL ------------------ #

    # plots_reverse_flux_and_TL(timelongitude, years)
    # plt.savefig(Meeting_notes_folder+model+'reverse_flux_and_TL.pdf', format="pdf", bbox_inches="tight")

    # # ------------------ Dipole tilt and TL ------------------ #

    # cut_TL, cut_years = timelongitude, years

    # plots_reverse_flux_dipole_tilt(cut_TL, cut_years)

    # # # # moving_window = np.loadtxt('/Users/nclizzie/Documents/Research/Time_Longitude/'+model+'/'+model+'_drift_MA_axisymmetric_HP.txt')
    # # # # mov_wind_years_cent = np.loadtxt('/Users/nclizzie/Documents/Research/Time_Longitude/'+model+'/'+model+'_drift_MA_axisymmetric_HP_MAyears.txt')
    # # # # max_signal = np.loadtxt('/Users/nclizzie/Documents/Research/Time_Longitude/'+model+'/'+model+'_drift_MA_axisymmetric_HP_MA_max_signal.txt')

    # plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_reverse_flux.pdf', format="pdf", bbox_inches="tight", dpi = 150)

    # # -------------------- Time-longitude and Radon transform -------------------- #

    # radon_image = Radon_Transformation(timelongitude, projection_degrees)
    # plot_TL_RT(timelongitude, years, dt, radon_image, drift)
    # drift_determ = drift_determination(radon_image, drift)
    # plot_drift_determination(drift_determ, drift, years, dt)

    # ------------------ Radon drift determination with latitude ------------------ #

    top_latitude, bottom_latitude   = 70 , -70          # 70 , -70 degrees north , south latitude
    latitude_spacing                = 2                # latitude degree spacing
    varying_latitude                = np.arange(top_latitude, bottom_latitude + - latitude_spacing, -latitude_spacing)


    # drift_latitude = np.loadtxt(TL_folder_Etote+model+'/'+model+'_drift_latitude_'+axisymm_state+'_HP'+str(freq_cut)+'.txt')

    drift_latitude, years = drift_determination_latitude(varying_latitude, drift, projection_degrees , year_start, year_end)
    # np.savetxt(TL_folder_Etote+model+'/'+model+'_drift_latitude_'+axisymm_state+'_HP'+str(freq_cut)+'.txt', drift_latitude)
    
    plot_drift_determination_latitude(drift_latitude, drift, varying_latitude, years , dt)
    plt.savefig(Meeting_notes_folder+model+'_drift_latitude_'+axisymm_state+'_HP'+str(freq_cut)+'.pdf', format="pdf", bbox_inches="tight", dpi = 150)


    # # ------------------ 2-D Fourier Transform ------------------ #

    # plot_wavenumber_frequency(timelongitude, dt, years)
    # plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_fourier2.pdf', format="pdf", bbox_inches="tight", dpi = 150)

    plt.show()


#################################### Subroutines ###########################################

def remove_axi_field_by_linear_reg(GC_Dict:dict, degree:int, year_start_index, year_end_index):
    """ Remove the zonal part of the field by linear regression
    """

    if   degree == 1:
        m_equals_zero = ['g1_0']
    elif degree == 2:
        m_equals_zero = ['g1_0', 'g2_0']
    elif degree == 3:
        m_equals_zero = ['g1_0', 'g2_0','g3_0']
    elif degree == 4:
        m_equals_zero = ['g1_0', 'g2_0','g3_0','g4_0']
    elif degree == 5:
         m_equals_zero = ['g1_0', 'g2_0','g3_0','g4_0','g5_0']
    elif degree == 6:
         m_equals_zero = ['g1_0', 'g2_0','g3_0','g4_0','g5_0','g6_0']
    elif degree == 7:
        m_equals_zero = ['g1_0', 'g2_0','g3_0','g4_0','g5_0','g6_0', 'g7_0']
    elif degree == 8:
        m_equals_zero = ['g1_0', 'g2_0','g3_0','g4_0','g5_0','g6_0', 'g7_0', 'g8_0']
    elif degree == 9:
        m_equals_zero = ['g1_0', 'g2_0','g3_0','g4_0','g5_0','g6_0', 'g7_0', 'g8_0', 'g9_0']
    elif degree == 10:
        m_equals_zero = ['g1_0', 'g2_0','g3_0','g4_0','g5_0','g6_0', 'g7_0', 'g8_0', 'g9_0', 'g10_0']
    else:
        print('Degree and Order not set')
        exit()

    reg_linear = linear_model.LinearRegression(fit_intercept=True)

    for m in m_equals_zero:
        m_linear_reg = reg_linear.fit(GC_Dict['year'][year_start_index:year_end_index].reshape((-1,1)), GC_Dict[m][year_start_index:year_end_index])

        plt.figure()
        plt.plot(GC_Dict['year'][year_start_index:year_end_index], GC_Dict[m][year_start_index:year_end_index],label = 'unaltered')
        plt.plot(GC_Dict['year'][year_start_index:year_end_index], m_linear_reg.predict(GC_Dict['year'][year_start_index:year_end_index].reshape((-1,1))),label = 'linear regression')

        GC_Dict[m][year_start_index:year_end_index] = GC_Dict[m][year_start_index:year_end_index]- m_linear_reg.predict(GC_Dict['year'][year_start_index:year_end_index].reshape((-1,1)))

        plt.plot(GC_Dict['year'][year_start_index:year_end_index], GC_Dict[m][year_start_index:year_end_index],label = 'altered')

        plt.legend()


def plot_dipole_moment(GC_Dict: dict, year_start, year_end):
    """Plots the dipole moment
    """

    plt.figure(figsize=(2.5,9))
    DM = np.sqrt(GC_Dict['g1_0']**2+GC_Dict['g1_1']**2+GC_Dict['h1_1']**2)

    plt.plot(DM, GC_Dict['ka'])
    plt.title(model + ' Dipole moment', fontsize = 15)

    if model == ("Monika_Pm10_Ra350" or "ledt002" or "ledt039"):
        plt.ylabel('Diffusion time', fontsize = 15)
    else:    
        plt.ylabel('Age (ka)', fontsize = 15)
    
    plt.xlabel('Dipole moment', fontsize = 10)

    plt.ylim(year_start, year_end)

def calculate_dipole_latitude_tilt(GC_Dict: dict):
    """Calculates the dipole tilt by latitude degrees using the Gauss coefficient dictionary
    """

    assert type(GC_Dict) == dict,       "GC_Dict is not a dictionary"
    # adjusted for latitude degrees paleomagnetism
    dipole_tilt = np.rad2deg( np.arccos( GC_Dict['g1_0']/ (np.sqrt( GC_Dict['g1_0']**2+ GC_Dict['g1_1']**2 + GC_Dict['h1_1']**2  )) ) )-90

    return dipole_tilt

def calculate_dipole_tilt(GC_Dict: dict):
    """Calculates the dipole tilt using the Gauss coefficient dictionary
    """

    assert type(GC_Dict) == dict,       "GC_Dict is not a dictionary"

    dipole_tilt = np.rad2deg( np.arccos( np.abs(GC_Dict['g1_0'])/ (np.sqrt( GC_Dict['g1_0']**2+ GC_Dict['g1_1']**2 + GC_Dict['h1_1']**2  )) ) )

    return dipole_tilt


def plot_dipole_tilt(GC_Dict: dict, year_start, year_end , degree_type = 'latitude'):
    """ Plots the dipole tilt either latitude or from the axis
    """
    
    assert type(GC_Dict) == dict,       "GC_Dict is not a dictionary"
    assert type(degree_type) == str,    "degree type is not a string"

    plt.figure(figsize=(2.5,9))
    if degree_type == 'latitude':
        dipole_tilt = calculate_dipole_latitude_tilt(GC_Dict)
        plt.plot(dipole_tilt, GC_Dict['ka'])
        plt.xlabel('Dipole tilt (latitude $\degree$)', fontsize= 10)
    
    if model == ("Monika_Pm10_Ra350" or "ledt002" or "ledt039"):
        plt.ylabel('Diffusion time', fontsize = 15)
    else:
        plt.ylabel('Age (ka)', fontsize = 15)
    
    plt.ylim(year_start, year_end)
    plt.title(model + ' Dipole tilt', fontsize = 15)


def plot_wavenumber_frequency(TL: np.ndarray, dt, years):
    """ Plots the 2D Fourier transform for the TL plot
    """

    kas_start = np.abs(years[0]-2000)/1000
    kas_end = np.abs(years[-1]-2000)/1000
    
    m = 360/(2*longitude_spacing)

    # padding the time-longitude plot
    TL_padded = np.pad(TL, int(2*len(TL)), mode='constant')
    print(TL_padded.shape)

    ff2 = np.fft.ifftshift(TL_padded)
    ff2 = np.fft.fft2(ff2)
    ff2 = np.fft.fftshift(ff2)

    plt.figure(figsize = (4,3))        
    
    # # flipud because the Fourier transform westward is postive and eastward is negative
    # plt.imshow(abs(np.fliplr(np.flipud(ff2)))/(1e2), aspect='auto', cmap= cm.devon.reversed(), \
    #     extent = [-m-0.5, m-0.5 , -1e3/(2*dt) ,1e3/(2*dt) ] )         # left, right, bottom, top. imshow is shifted by half

    plt.imshow(abs(np.rot90(ff2))/(1e3), aspect='auto', cmap= cm.devon.reversed(), \
            vmin=0, vmax = 8.5, \
            extent = [-1e3/(2*dt) ,1e3/(2*dt), -m-0.5, m-0.5 ] )         # left, right, bottom, top. imshow is shifted by half

    plt.ylim([0,  4])
    plt.xlim(-1, 1)         # Note, order is reversed for x
    plt.xlabel('WW  Frequency ($10^{-3}yr^{-1}$) EW', fontsize = 15)
    plt.ylabel('Wavenumber', fontsize= 15)

    plt.title(model+'_p: ' + str(kas_end) + " to " + str(kas_start) + 'ka'+ str(latitude)+"\n", fontsize= 10)
    plt.colorbar(label= 'Spectral power ($10^{3}mT^2$)', location = 'right')



def reverse_flux_from_TL(TL: np.ndarray):
    """  Sums the reverse flux patches across the longitudinal axis on the time-longitude plot
    """

    TL_max_5percent = np.mean(TL)*0.1
    print('TL 10% mean: ', TL_max_5percent)
    reverse_flux = np.where(TL >= TL_max_5percent)
    reverse_TL = np.zeros(TL.shape)
    reverse_TL[reverse_flux[0],reverse_flux[1]] = TL[reverse_flux[0], reverse_flux[1]]
    reverse_flux_time = np.abs(reverse_TL).sum(axis=1)

    return reverse_flux_time

def plots_reverse_flux_and_TL(TL: np.ndarray, years: np.ndarray):
    """ Plot the reverse flux patches summation on the left and time-longitude on the right
    """

    kas = np.abs(years-2000)/1000
    print(kas[-1], kas[0])

    if model == "GGF100k":
        fig     = plt.figure(figsize= (6, 15))    # figsize = width, height
        # fig     = plt.figure(figsize= (8, 8))    # figsize = width, height
    elif model == "GGFSS70":
        fig     = plt.figure(figsize= (5.2, 6.825))    # figsize = width, height
    elif model == "LSMOD.2":
        fig     = plt.figure(figsize= (5.5, 2.475))    # figsize = width, height

    grid    = fig.add_gridspec(nrows = 1, ncols= 2, width_ratios = (1, 6),
                               left = 0.1, right = 0.9, bottom = 0.1, top = 0.9,
                               wspace = 0.11)      # GGF100k 0.065, GGFSS70 0.11  , LSMOD.2  0.11 

    # Reverse flux summation (left panel)
    ax00 = fig.add_subplot(grid[0])
    ax00.plot(reverse_flux_from_TL(TL), kas, 'darkred')
    ax00.set_ylim(kas[-1], kas[0])

    if model == "GGF100k":
        ax00.set_yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
        ax00.axhspan(16.5, 17.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(32.5, 33.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(40.5, 41.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(60.5, 61.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(94.5, 95.5 , facecolor= 'gray', alpha = 0.4)
        fontsize_word = 20
        fontsize_tick = 15
        fontsize_small_word = 15
        ax00.set_ylabel('Age (ka)', fontsize = fontsize_word)
        # ax00.set_xticks([0, 5])

    elif model == "GGFSS70":
        ax00.set_yticks([15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
        ax00.axhspan(16.5, 17.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(32.5, 33.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(40.5, 41.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(60.5, 61.5 , facecolor= 'gray', alpha = 0.4)
        fontsize_word = 16
        fontsize_tick = 12
        fontsize_small_word = 12
        ax00.set_ylabel(' ', fontsize = fontsize_small_word)
        ax00.set_xticks([0, 10])

    elif model == "LSMOD.2":
        ax00.set_yticks([30, 35, 40, 45, 50])
        ax00.axhspan(32.5, 33.5 , facecolor= 'gray', alpha = 0.4)
        ax00.axhspan(40.5, 41.5 , facecolor= 'gray', alpha = 0.4)
        fontsize_word = 15
        fontsize_tick = 12
        fontsize_small_word = 12
        ax00.set_ylabel(' ', fontsize = fontsize_word)
        ax00.set_xticks([0, 20])

    ax00.set_xlabel('Reverse/weak\nflux |mT|', fontsize= fontsize_word)
    ax00.tick_params(axis='x', labelsize = fontsize_tick)
    ax00.tick_params(axis='y', labelsize =fontsize_tick)    

    ax00.tick_params(width=0.25)

    # Time-longitude plot (right panel)
    ax01 = fig.add_subplot(grid[1])
    
    TL_show = ax01.imshow(TL, aspect='auto', cmap=cm.vik, \
            # vmin   = -np.max(np.abs(TL)) ,vmax= np.max(np.abs(TL)), 
            # vmin = -1.1 ,vmax= 1.1,                       # 0.8126             
            vmin   = -0.8126 ,vmax = 0.8216, 
            extent = [longitude_start, longitude_end, kas[-1], kas[0]])

    print('max: ',  np.max(TL), 'min: ', np.min(TL))
    
    ax01.set_title(model + ' at ' + str(latitude), fontsize= 10)
    ax01.set_xlabel('Longitude ($^{\circ}$E)', fontsize= fontsize_word)
    ax01.set_xticks([-180, -90, 0, 90, 180])
    ax01.set_yticks([])
    ax01.tick_params(width=0.25)
    ax01.tick_params(axis='x', labelsize = fontsize_tick)

    for axis in ['top','bottom','left','right']:
        ax00.spines[axis].set_linewidth(0.25)
        ax01.spines[axis].set_linewidth(0.25)

    # if      model == "GGF100k":
    #     cbar = plt.colorbar(TL_show, aspect = 27)
    #     # cbar = plt.colorbar(TL_show, aspect = 20)
    #     # cbar.ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    #     # ax01.axvline(120, color = 'gray', linestyle = '-')    # linestyle = '-'
    # elif    model == "GGFSS70":
    #     cbar = plt.colorbar(TL_show, aspect = 25)
    # elif    model == "LSMOD.2":
    #     cbar = plt.colorbar(TL_show, aspect = 15)

    # if milliTesla == "mT_on":
    #     cbar.set_label(label = '$B_r$ (mT)', fontsize = fontsize_word)
    # else:
    #     cbar.set_label(label = '$B_r$ (nT)', fontsize = fontsize_word)
    
    # cbar.ax.tick_params(labelsize= fontsize_tick, rotation = 0)

def save_TL_fig():
    """ Saves the time-longitude figure and saves the time-longitude as a text document
    """

    assert type(date) and type(model)                  == str          , 'date or model is not a string'
    assert type(high_pass) and type(axisymm_state)     == str          , 'high_pass, axisymmetric, or milliTesla is not a str'

    if      (high_pass == "HP_on") and (axisymm_state == "axi_off"):
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_HP'+str(freq_cut)+'_nonaxisym.pdf', format="pdf", bbox_inches="tight")
    elif    (high_pass == "HP_on") and (axisymm_state == "axi_on"):
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_HP'+str(freq_cut)+'_allcoeff.pdf', format="pdf", bbox_inches="tight")
    elif    (high_pass == "HP_off") and (axisymm_state == "axi_off"):
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_nonaxisym.pdf', format="pdf", bbox_inches="tight")
    elif    (high_pass == "HP_off") and (axisymm_state == "axi_on"):
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_allcoeff.pdf', format="pdf", bbox_inches="tight")
    else:
        print('Sayonara Sammy')


def cut_timelongitude(timelongitude: np.ndarray, years:np.ndarray, younger_year, older_year):
    """Cuts the time-longitude plot based off of the desired years
    """

    assert type(timelongitude) and type(years) == np.ndarray
    assert type(younger_year) and type(older_year) == int or np.float64
    assert years[0] > years[-1],        'The years and time-longitude needs to be flipped'
    assert younger_year > older_year,      'The top year on the time-longitude plot is not greater than the lowest year'
    assert timelongitude.shape[0] == years.shape[0]
    assert younger_year <= years[0],        'the top year is out of bounds for the time-longitude plot'
    assert older_year >= years[-1],    'the bottom year is out of bounds for the time-longitude plot'

    # find the index in the year array
    younger_index       = (np.abs(years - younger_year)).argmin()
    older_index         = (np.abs(years - older_year)).argmin()+1

    # cut the time-longitude and years
    cut_TL = timelongitude[younger_index:older_index, :]
    cut_years = years[younger_index:older_index]
    cut_x_pixels = cut_TL.shape[1]
    cut_y_pixels = cut_TL.shape[0]

    assert cut_y_pixels == cut_years.shape[0]

    return cut_TL, cut_years, cut_x_pixels, cut_y_pixels


def drift_determination_latitude(varying_latitude, drift, projection_degrees , year_start, year_end):
    """ Drift determination with varying latitude
    """

    assert varying_latitude[0] > varying_latitude[-1], 'most northern latitude is not above the most southern latitude'
    assert type(drift) and type(varying_latitude) and type(projection_degrees) == np.ndarray,   'drift, varying_latitude, or projection_degrees are not np.ndarray'
    assert type(milliTesla) == str,    'millTesla is not a str'
    assert type(HP_order) == int,       'HP_order is not an int'

    # Initializing drift_latitude matrix
    drift_latitude = np.zeros([len(varying_latitude), len(drift)])

    j = 0
    for i in varying_latitude:

        timelong_latitude, years_com, x_pixels, y_pixels    = time_longitude.main(model, axisymm_state, high_pass, year_start, year_end, freq_cut_faux=freq_cut, latitude_faux = i)
        timelong_latitude, years_com                        = check_TL_order(timelong_latitude, years_com)
        if high_pass == "HP_on":
            timelong_latitude, years_com, x_pixels, y_pixels    = cut_off_edges_HP_filter(timelong_latitude, years_com)

        radon_image_lat = Radon_Transformation(timelong_latitude, projection_degrees)
        drift_determ_lat = drift_determination(radon_image_lat, drift)
        drift_latitude[j, :] = drift_determ_lat
        j = j +1

        print('Latitude: ', i)
        print('The size of the years; ', len(years_com) )
        print('Drift determintation latitude years: ' ,years_com[-1], years_com[0])

    return drift_latitude, years_com


def plot_drift_determination_latitude(drift_latitude, drift, varying_latitude, years , dt):
    """ Plots the drift determination with varying latitude
    """

    assert type(drift_latitude) and type(drift) and type(varying_latitude) and type(years) == np.ndarray
    assert type(dt) == np.float64,      'dt is not a float'
    assert type(model) == str,          'model is not a string'

    kas = np.abs(years-2000)/1000
    fontsize_word = 18                  # pfms: 18
    fontsize_tick = 12.5                # pfms: 12.5
    fontsize_small_word = 18            # pfms: 16

    fig = plt.figure(figsize= (5,5))
    grid    = fig.add_gridspec(nrows = 1, ncols= 1,
                               left = 0.1, right = 0.9, bottom = 0.1, top = 0.9)
    ax  = fig.add_subplot(grid[0])

    # left, right, bottom, top for extend
    Drift_lat = ax.imshow(drift_latitude/(1e6), aspect='auto', cmap = cm.oslo.reversed(), \
                          vmin = 0, vmax = np.max(drift_latitude/(1e6)), \
                          extent=[drift[0]/(1e4), drift[-1]/(1e4), varying_latitude[-1], varying_latitude[0]])
    
    print(np.max(drift_latitude/(1e6)))

    plt.plot([-29000/(1e4), -30000/(1e4)], [60, 60],  c= "royalblue", linewidth = 4.5)
    plt.plot([27400/(1e4), 28400/(1e4)], [60, 60],   c= "teal",      linewidth = 4.5)
    plt.plot([-17500/(1e4), -18500/(1e4)], [60, 60], c= "deeppink",  linewidth = 4.5)

    if model == "Monika_Pm10_Ra350" or "ledt002" or "ledt039":
        ax.set_xlabel('WW Drift rate ($10^{4}$ $^{\circ}/t_d$) EW', fontsize = fontsize_word)
    else:
        ax.set_xlabel('WW  Drift rate ($10^{4} \degree /year$)  EW', fontsize= fontsize_word)
    ax.set_ylabel('South   Latitude ($^{\circ}$)   North', fontsize= fontsize_word)
    # plt.xticks([-0.3, -0.2, -0.1 , 0, 0.1, 0.2, 0.3])
    ax.set_yticks([70, 60, 50, 40, 30, 20, 10, 0, -10, -20, -30, -40, -50, -60, -70 ])
    ax.tick_params(axis='x', labelsize = fontsize_tick)
    ax.tick_params(axis='y', labelsize = fontsize_tick)

    ax.axhline(0, color = 'gray', linestyle = '-', linewidth = 0.5)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.25)

    # plt.colorbar(label = '$ 10^{5}nT^2$')
    ax.set_title(model + ': ' + f"{years[-1]:.3f}" + ' to ' + f"{years[0]:.3f}" + ' ka\n', fontsize= 10)
    
    cbar = plt.colorbar(Drift_lat, aspect = 30)
    cbar.set_label('Drift signal power ($10^{6}$)', fontsize = fontsize_small_word)
    cbar.ax.tick_params(labelsize= fontsize_tick)
    cbar.outline.set_linewidth(0.25)
    cbar.ax.tick_params(width = 0.25)


def load_TL_doc(TL_folder:str):
    """ Loads the time-longitude data from the directory in .../Research
    """

    # "axi_keep" "axi_remove_TA" "axi_remove_lin_reg"

    assert type(model) == str and type(high_pass) and type(axisymm_state) == str,          'model, high_pass, or axisym_state is not a string'

    if model == "model_test":
        years_complete          = np.loadtxt('/Users/nclizzie/Documents/Research/Time_Longitude/'+model+'/'+model+'_years_for_TL.txt')
        timelongitude_complete  = np.loadtxt('/Users/nclizzie/Documents/Research/Time_Longitude/'+model+'/'+model+'_TL.txt')


    elif (high_pass == "HP_on") and (axisymm_state == "axi_remove_TA"):
        years_complete           = np.loadtxt(TL_folder+model+'/'+model+'_years_for_TL_HP'+str(freq_cut)+'_Axi_TA_removed.txt')
        timelongitude_complete   = np.loadtxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_Axi_TA_removed.txt')

    elif (high_pass == "HP_on") and (axisymm_state == "axi_remove_lin_reg"):
        years_complete           = np.loadtxt(TL_folder+model+'/'+model+'_years_for_TL_HP'+str(freq_cut)+'_Axi_removed_byLR.txt')
        timelongitude_complete   = np.loadtxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_Axi_removed_byLR.txt')

    elif (high_pass == "HP_on") and (axisymm_state == "axi_keep"):
        years_complete           = np.loadtxt(TL_folder+model+'/'+model+'_years_for_TL_HP'+str(freq_cut)+'_allcoeffs.txt')
        timelongitude_complete   = np.loadtxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_allcoeffs.txt')

    elif (high_pass == "HP_off") and (axisymm_state == "axi_remove_TA"):
        years_complete           = np.loadtxt(TL_folder+model+'/'+model+'_years_for_TL_Axi_TA_removed.txt')
        timelongitude_complete   = np.loadtxt(TL_folder+model+'/'+model+'_TL_Axi_TA_removed.txt')

    elif (high_pass == "HP_off") and (axisymm_state == "axi_remove_lin_reg"):
        years_complete           = np.loadtxt(TL_folder+model+'/'+model+'_years_for_TL_Axi_removed_byLR.txt')
        timelongitude_complete   = np.loadtxt(TL_folder+model+'/'+model+'_TL_Axi_removed_byLR.txt')

    elif (high_pass == "HP_off") and (axisymm_state == "axi_keep"):
        years_complete           = np.loadtxt(TL_folder+model+'/'+model+'_years_for_TL_allcoeffs.txt')
        timelongitude_complete   = np.loadtxt(TL_folder+model+'/'+model+'_TL_allcoeffs.txt')

    else:
        print('Sayonara Sammy')

    x_pixel = timelongitude_complete.shape[1]            # longitude pixel in time longitude plot
    y_pixel = timelongitude_complete.shape[0]            # time pixel in time longitude plot
    assert timelongitude_complete.shape[0] == years_complete.shape[0], 'Time-longitude and years are not the same size'

    return timelongitude_complete, years_complete, x_pixel, y_pixel


def check_TL_order(timelongitude_complete, years_complete):
    """ This checks to make sure the timelongitude plot is in the correct order. Youngest at the top and oldest at the bottom
    """

    assert type(timelongitude_complete) and type(years_complete) == np.ndarray, 'timelongitude_complete and years_complete are not np.ndarray'
    assert timelongitude_complete.shape[0] == years_complete.shape[0],  'timelongitude_complete and years_complete are the same row size'

    print('The first/top index of years: ', years_complete[0])
    print('The last/bottom index of years: ', years_complete[-1])

    if years_complete[0] < years_complete[-1]:
        timelongitude_complete = np.flipud(timelongitude_complete)
        years_complete = np.flipud(years_complete)
        print('The years were reverse, first/top index now: ', years_complete[0])
        print('The years were reverse, last/bottom index now: ', years_complete[-1])

    return timelongitude_complete, years_complete


def cut_off_edges_HP_filter(timelongitude_complete, years_complete):
    """ This cuts off the edges due to high pass filter
    """

    assert type(timelongitude_complete) and type(years_complete) == np.ndarray,     'not np.darray'
    assert type(HP_order) == int,    'high_pass_order is not an int'
    assert timelongitude_complete.shape[0] == years_complete.shape[0], 'Time-longitude and years are not the same size'

    print('Time longitude size before edge removal: ', timelongitude_complete.shape)
    print('Before removal: first index year: ', years_complete[0], ' and the last index year: ', years_complete[-1])
    timelongitude = timelongitude_complete[HP_order:timelongitude_complete.shape[0]-HP_order, :]
    years         = years_complete[HP_order:years_complete.shape[0]- HP_order]
    print('Time longitude size after edge removal  : ', timelongitude.shape)
    print('After removal: first index year: ', years[0], ' and the last index year: ', years[-1])

    x_pixels      = timelongitude.shape[1]
    y_pixels      = timelongitude.shape[0]

    assert y_pixels == years.shape[0], 'Time-longitude and years are not the same size'

    return timelongitude, years, x_pixels, y_pixels


def convert_nT_to_mT(timelongitude):

    assert type(timelongitude) == np.ndarray,   'timelongitude is not an np.ndarray'

    timelongitude = timelongitude/(1e6)
    print('The time longtiude has been converted from nT to mT')

    return timelongitude


def Radon_Transformation(timelongitude, projection_degrees):
    """ Radon transformation
    """

    assert type(timelongitude) and type(projection_degrees) == np.ndarray

    radon_image = skimage.transform.radon(timelongitude, theta = projection_degrees, circle = False, preserve_range = True)

    return radon_image


def plot_TL_RT(timelongitude, years, dt, radon_image, drift):
    """Plot time longitude and radon transform plot
    """

    assert type(timelongitude) and type(years) and type(radon_image) and type(drift) == np.ndarray
    assert type(model) and type(milliTesla) == str,          'model or milliTesla are not a string'
    assert type(latitude) and type(dt) == int or np.float64,       'HP_order is not an int'

    # Left panel showing the time longitude plot

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(timelongitude, aspect='auto', cmap= cm.vik, \
        vmin = -np.max(np.abs(timelongitude)), vmax= np.max(np.abs(timelongitude)), \
        extent= [longitude_start, longitude_end, years[-1], years[0]] )

    # Label colorbar in mT or nT
    if milliTesla == "mT_on":
        plt.colorbar(label = '$B_r$ (mT)')
    else:
        plt.colorbar(label = '$B_r$ (nT)')

    # print x ticks (longitude) on TL
    plt.xticks([-180, -90, 0, 90, 180])

    plt.xlabel('Longitude ($^{\circ}$ E)', fontsize= 10)
    if model == ("Monika_Pm10_Ra350" or "ledt002" or "ledt039"):
        plt.ylabel('Diffusion time', fontsize = 10)
    else:
        plt.ylabel('Year', fontsize = 10)
    plt.title('Time Longitude for '+ model + ' latitude: ' + str(latitude), fontsize = 15)

    # ----------------------------------------------------------------------------------

    # Right panel showing the radon transform

    plt.subplot(1,2,2)
    plt.imshow(radon_image, aspect= 'auto', cmap= cm.vik, \
        vmin = -np.max(np.abs(radon_image)) ,vmax= np.max(np.abs(radon_image)), \
        extent= [drift[0], drift[-1], 0, radon_image.shape[0]] )

    # Label colorbar in mT or nT
    if milliTesla == "mT_on":
        plt.colorbar(label = 'mT $years/^{o}$')
    else:
        plt.colorbar(label = 'nT $years/^{o}$')

    # don't printing y ticks (pixel) on RT

    # print x ticks (longitude) on TL
    # plt.xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])


    if model == ("Monika_Pm10_Ra350" or "ledt002" or "ledt039"):
        plt.xlabel('Drift rate ($^{o}/diffusion time$)', fontsize = 10)
    else:
        plt.xlabel('Drift rate ($^{o}$/year)', fontsize = 10)
    plt.ylabel('Longitude pixel on TL', fontsize = 10)
    plt.title(model + ' Radon Transformation')


def drift_to_projection(drift_low, drift_high, drift_inc, x_pixel, dt):
    """ Calcualtes the projection angle from drift.
    Output projection angle is from both negative and positive drift
    """

    assert type(drift_low) and type(drift_high) and type(drift_inc) and type(x_pixel) and type(dt) == int or np.float64

    drift               = np.arange(drift_low, drift_high+drift_inc, drift_inc)
    theta_degrees       = np.rad2deg( np.arctan(360/(drift*x_pixel*dt)) )

    projection_degrees  = np.concatenate( (np.flipud((90-theta_degrees)),-(90-theta_degrees) ) )

    print('Projection start: ', projection_degrees[0], ' and projection end:', projection_degrees[-1])

    return projection_degrees

def driftrate_to_velocity(drift:np.ndarray):
    """ Calculates the drift rate (latitude degrees/year) to velocity (km/year) at the CMB
    """

    R_cmb       = 3483                                          # Radius of outer core in km
    R_lat       = R_cmb*np.cos(np.deg2rad(np.abs(latitude)))    # Radius of the small circle at the latitude
    circ_lat    = 2*np.pi*R_lat                                 # Circumference of the small circle

    velocity = drift*circ_lat/360
    print('Circumference of latitudinal small circle: ',circ_lat)

    return velocity


def moving_window_RT(timelongitude_complete, years_complete, year_increments, year_window, projection_degrees, dt, drift, normalization = 'no normalization'):
    """ Does the loop to break down the time-longitude into windows and performs the Radon transform
    """

    assert type(timelongitude_complete) and type(years_complete) and type(projection_degrees) and type(drift) == np.ndarray
    assert type(year_increments) and type(year_window) and type(dt) and type(latitude) == int or np.float64
    assert type(model) and type(milliTesla) == str,                         'model or milliTesla are not a string'
    assert timelongitude_complete.shape[0] == years_complete.shape[0],      'Time-longitude and years are not the same size'
    assert projection_degrees.shape[0] == drift.shape[0],                   'Projection_degrees and drift are not the same size'
    assert normalization == 'time step' or 'whole time' or 'no normalization'
    assert years_complete[0] > years_complete [-1],                         'years might be flipped'

    mov_wind_years_start = np.arange(years_complete[0], years_complete[-1]+year_window, -year_increments)
    mov_wind_years_cent  = mov_wind_years_start - year_window/2

    moving_window = np.zeros([len(mov_wind_years_start), len(drift)])
    print('moving window row size: ', moving_window.shape[0], ' and column size: ', moving_window.shape[1])

    max_signal          = np.zeros(len(mov_wind_years_start))

    val = 360/year_window
    indx1 = np.abs(drift+val).argmin()
    indx2 = np.abs(drift-val).argmin()

    j = 0
    for i in mov_wind_years_start:

        # index location in the year array
        # locations = int(np.where(i==years_complete)[0])
        locations = np.abs(years_complete-i).argmin()

        # The years of the time-longitude the Radon transform will be performed
        print(years_complete[locations], ' to ', years_complete[locations+int(year_window/dt)])

        # Sectioning out the time-longitude plot for the desired years
        timelongitude_window   = timelongitude_complete[locations : locations+int(year_window/dt),:]

        radon_image = Radon_Transformation(timelongitude_window, projection_degrees)
        drift_determ = drift_determination(radon_image, drift)

        if np.max(drift_determ) == 0:
            moving_window[j, :] = drift_determ
            print('No signal in the radon transform')

        if normalization == 'time step':
            # moving_window[j, :] = drift_determ/np.max(drift_determ[np.r_[0:indx1,indx2:-1]])
            moving_window[j, :] = drift_determ/np.max(drift_determ)
            print('Normalizing moving average by each time step')
        elif normalization == 'whole time':
            moving_window[j, :] = drift_determ
        else:
            moving_window[j, :] = drift_determ
            print('No normalization')

        # max_signal[j] = np.max(drift_determ[np.r_[0:indx1,indx2:-1]])
        max_signal[j] = np.max(drift_determ)
        j = j+1

    if normalization == 'whole time':
        moving_window = moving_window/np.max(max_signal)
        print('Normalizing moving average by entire series')

    return moving_window, mov_wind_years_cent, max_signal


def drift_determination(radon_image, drift):
    """ Collapsing the radon transformation
    """

    assert type(radon_image) and type(drift) == np.ndarray,     'radon_image or drift are not np.ndarray'

    drift_determ = np.sum( np.square(radon_image), axis=0 )

    assert drift_determ.shape[0] == drift.shape[0]

    return drift_determ


def plot_drift_determination(drift_determ, drift, years, dt):
    """ Plots drift determination for one radon drift determination
    """

    assert type(drift_determ) and type(drift)   == np.ndarray,     'radon_image or drift are not np.ndarray'
    assert type(milliTesla)                     == str,            'millTesla is not a string'

    plt.figure()
    plt.plot(drift, drift_determ )

    # Line below for marking a drift line
    # plt.axvline(-0.09, color = 'b', linestyle = '-.')

    # Gray block out of high period or low frequency, this depends on the length of the time series
    plt.axvline(-360/(len(years)*dt), color = 'darkgrey')
    plt.axvline( 360/(len(years)*dt), color = 'darkgrey')
    # plt.axvspan(-360/(len(years)*dt), 360/(len(years)*dt) , facecolor = '0.75')
    print('Blocking out: '+ str(-360/(len(years)*dt)) + ' to ' + str(360/(len(years)*dt)))

    if model == ("Monika_Pm10_Ra350" or "ledt002" or "ledt039"):
        plt.xlabel('Drift rate ($^{o}/diffusion time$)')
    else:
        plt.xlabel('Drift rate ($^{o}$/year)', fontsize = 15)
    
    if milliTesla == "mT_on":
        plt.ylabel('Radon drift ($mT^{2}$$(years/^{o})^2$)', fontsize = 15)
    else:
        plt.ylabel('Radon drift ($nT^{2}$$(years/^{o})^2$)', fontsize = 15)

    plt.title('Drift for '+ model+ ' from '+ str(years[0]) +' to '+ str(years[-1]))

def calculate_ADM(GC_Dict: dict):
    """Calculates the axial dipole moment
    """

    radius = 6371000                                            # Earth's radius in meters
    ADM = ((radius)**3*1e7)*np.abs(GC_Dict['g1_0']*1e-9)        # convert g10 from nT to T

    return ADM

def plot_moving_window_and_max_signal_and_tilt_and_moment(GC_Dict: dict, moving_window, max_signal, years, drift, mov_wind_years_cent, year_increments, year_window, dt):
    """Plot axial dipole moment with dipole tilt (left panel), max signal (middle panel), moving average RD (right panel)
    """

    assert type(GC_Dict) == dict
    assert type(moving_window) and type(drift) and type(mov_wind_years_cent) == np.ndarray
    assert type(model) and type(milliTesla) == str,          'model or milliTesla are not strings'
    assert type(year_increments) and type(year_window) and type(dt) == int or np.float64

    fontsize_word = 25
    fontsize_tick = 20

    if model == 'ledt002':
        mov_wind_ka_cent = mov_wind_years_cent

        fig  = plt.figure(figsize=(6.2, 7.48))    # width, height
        grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                            left = 0.15, right = 0.9, bottom = 0.1, top = 0.9
                            )
        
        ax011 = fig.add_subplot(grid[0])         # dipole tilt
        ax011a = ax011.twiny()

        # moving window (MW)
        MW = ax011.imshow(moving_window/(1e6), aspect='auto', cmap= cm.lajolla, \
                    # vmin = 0 ,vmax = 3.1734, \
                    vmin = 0 ,vmax = np.max(moving_window/(1e6)), \
                    extent= [drift[0]/(1e4), drift[-1]/(1e4), mov_wind_ka_cent[-1], mov_wind_ka_cent[0]])
        
        print('max moving window: ', np.max(moving_window/(1e6)))
        
        ax011.plot([-30000/(1e4), -30000/(1e4)], [0.94, 1.05],  ls="--", c= "royalblue", linewidth = 2)
        ax011.plot([27900/(1e4), 27900/(1e4)], [0.87, 0.93],    ls="--", c= "teal",      linewidth = 2)
        ax011.plot([-18000/(1e4), -18000/(1e4)], [0.75, 0.84],  ls="--", c= "deeppink",  linewidth = 2)

        ax011.set_xlabel('westward  Drift rate ($10^{4}$ $^{\circ}/t_d$)  eastward', fontsize= 18)
        ax011.set_ylabel('MA centered diffusion time', fontsize= 18)
        ax011.set_title(model+"_p moving average "+str(latitude)+"$\degree$ \n", fontsize= 10)
        ax011.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax011a.set_xlim(driftrate_to_velocity(drift)[0]/(1e6), driftrate_to_velocity(drift)[-1]/(1e6))
        ax011a.set_xlabel('Velocity ($10^{6}$ $km/t_d$)', fontsize= 18)
        ax011.tick_params(axis='x', labelsize = 12.5)
        ax011a.tick_params(axis='x', labelsize = 12.5)
        ax011.tick_params(axis='y', labelsize = 12.5)
        
        cbar = plt.colorbar(MW, aspect = 35)
        cbar.set_label('Drift signal power ($10^{6}$)', fontsize = 18)
        cbar.ax.tick_params(labelsize=12.5)



    elif model == 'GGF100k':
        mov_wind_ka_cent = np.abs(mov_wind_years_cent-2000)/1000

        fig  = plt.figure(figsize=(14, 18))    # width, height
        grid = fig.add_gridspec(nrows = 1 , ncols= 2, width_ratios = (1, 10),
                                left = 0.1, right = 0.9, bottom = 0.1, top = 0.9,
                                wspace = 0.18)


        # Dipole tilt and axial dipole moment
        grid00 = grid[0].subgridspec(nrows=1, ncols=1)

        color1, color2 = 'black' , 'darkred'
        ax000 = fig.add_subplot(grid00[0])         # dipole tilt
        ax001 = ax000.twiny()                      # axial dipole moment

        ax000.plot(calculate_dipole_tilt(GC_Dict), GC_Dict['ka'], color1)
        ax001.plot(calculate_ADM(GC_Dict)/(1e21), GC_Dict['ka'], color2)

        ax000.set_xlabel('Dipole\ntilt ($\degree$)', fontsize = fontsize_word, color= color1)
        ax000.set_ylabel('Age (ka)', fontsize = fontsize_word)
        ax000.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax000.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])        # GGF100k
        ax000.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax000.tick_params(axis= 'y', labelsize = fontsize_tick)

        ax001.set_xlabel('ADM ($ZAm^2$)', fontsize= fontsize_word,  color=color2)
        ax001.set_xlim(100,0)
        ax001.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax001.tick_params(axis= 'y', labelsize = fontsize_tick)

        # Maximum drift signal and moving window
        grid01 = grid[1].subgridspec(nrows=1, ncols=2, width_ratios = (1, 9.5),
                                wspace = 0.05)

        ax010  = fig.add_subplot(grid01[0,0])        # max drift signal
        ax011  = fig.add_subplot(grid01[0,1])        # moving window (MW)
        ax011a = ax011.twiny()
        ax011.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax011a.tick_params(axis= 'x', labelsize = fontsize_tick)


        ax010.plot(max_signal/(1e4), mov_wind_ka_cent, 'k')
        ax010.set_xlabel('Max signal\npower ($10^{4}$)', fontsize= fontsize_word)
        ax010.set_ylabel('MA centered age', fontsize= fontsize_word)
        ax010.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax010.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])        # GGF100k
        ax010.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax010.tick_params(axis= 'y', labelsize = fontsize_tick)
        
        # # moving window (MW)
        # if latitude == 55:
        #     vmax_55NorS = 15
        #     ax010.set_xticks([0, 7])
        # if latitude == -55:
        #     vmax_55NorS = 15
        #     ax010.set_xticks([0, 10])

        # moving window (MW)
        MW = ax011.imshow(moving_window/(1e4), aspect='auto', cmap= cm.lajolla, \
                        # vmin = 0 ,vmax = vmax_55NorS, \
                        vmin = 0 ,vmax = np.max(moving_window/(1e4)), \
                        extent= [drift[0], drift[-1], mov_wind_ka_cent[-1], mov_wind_ka_cent[0]])
    
        print('Max drift moving average signal:', np.max(moving_window))

        ax011.set_xlabel('WW   Drift rate ($\degree /year$)  EW', fontsize= fontsize_word)
        ax011.set_title(model+"_p moving average "+str(latitude)+"$\degree$\n", fontsize= 10)
        ax011.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax011.set_yticks([])

        color_1, color_2, color_3 = "royalblue" , "teal", "deeppink"
        # ax011.plot([-0.08, -0.08], [32, 39],  ls="--", c=color_3, linewidth = 1.5)
        ax011.plot([-0.07, -0.07], [32, 39],  ls="--", c=color_3, linewidth = 1.5)
        ax011.plot([0.11, 0.11], [40, 50],  ls="--", c=color_1, linewidth = 1.5)
        ax011.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        ax011a.set_xlim(driftrate_to_velocity(drift)[0], driftrate_to_velocity(drift)[-1])
        ax011a.set_xlabel('Velocity ($km/year$)', fontsize= fontsize_word)

        cbar = plt.colorbar(MW)
        cbar_label = [0, 5, 10, 15]
        cbar.set_label('Drift signal power ($10^{4}$)', fontsize = fontsize_word)
        cbar.ax.tick_params(labelsize= fontsize_tick, rotation = 90)

    elif model == 'GGFSS70':
        mov_wind_ka_cent = np.abs(mov_wind_years_cent-2000)/1000

        fig  = plt.figure(figsize=(14, 9.42))    # width, height. GGFSS70: 14, 4.65
        grid = fig.add_gridspec(nrows = 1 , ncols= 2, width_ratios = (1, 10),
                                left = 0.1, right = 0.9, bottom = 0.1, top = 0.9,
                                wspace = 0.18)


        # Dipole tilt and axial dipole moment
        grid00 = grid[0].subgridspec(nrows=1, ncols=1)

        color1, color2 = 'black' , 'darkred'
        ax000 = fig.add_subplot(grid00[0])         # dipole tilt
        ax001 = ax000.twiny()                      # axial dipole moment

        ax000.plot(calculate_dipole_tilt(GC_Dict), GC_Dict['ka'], color1)
        ax001.plot(calculate_ADM(GC_Dict)/(1e21), GC_Dict['ka'], color2)

        ax000.set_xlabel('Dipole\ntilt ($\degree$)', fontsize = fontsize_word, color= color1)
        ax000.set_ylabel('Age (ka)', fontsize = fontsize_word)
        ax000.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax000.set_yticks([20, 30, 40, 50, 60])
        ax000.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax000.tick_params(axis= 'y', labelsize = fontsize_tick)

        ax001.set_xlabel('ADM ($ZAm^2$)', fontsize= fontsize_word,  color=color2)
        ax001.set_xlim(100,0)
        ax001.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax001.tick_params(axis= 'y', labelsize = fontsize_tick)

        # Maximum drift signal and moving window
        grid01 = grid[1].subgridspec(nrows=1, ncols=2, width_ratios = (1, 9.5),
                                wspace = 0.05)

        ax010 = fig.add_subplot(grid01[0,0])        # max drift signal
        ax011 = fig.add_subplot(grid01[0,1])        # moving window (MW)
        ax011a = ax011.twiny()                      # drift velocity
        ax011.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax011a.tick_params(axis= 'x', labelsize = fontsize_tick)

        ax010.plot(max_signal/(1e4), mov_wind_ka_cent, 'k')
        ax010.set_xlabel('Max signal\npower ($10^{4}$)', fontsize= fontsize_word)
        ax010.set_ylabel('MA centered age', fontsize= fontsize_word)
        ax010.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax010.set_yticks([20, 30, 40, 50, 60])
        ax010.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax010.tick_params(axis= 'y', labelsize = fontsize_tick)

        # moving window (MW)
        if latitude == 55:
            vmax_55NorS = 15
            ax010.set_xticks([0, 4])
        if latitude == -55:
            vmax_55NorS = 15
            ax010.set_xticks([0, 20])
        
        # moving window (MW)
        MW = ax011.imshow(moving_window/(1e4), aspect='auto', cmap= cm.lajolla, \
                        vmax = 0, vmin = vmax_55NorS, \
                        extent= [drift[0], drift[-1], mov_wind_ka_cent[-1], mov_wind_ka_cent[0]])

        print('Max drift moving average signal:', np.max(moving_window))

        ax011.set_xlabel('WW   Drift rate ($\degree /year$)   EW', fontsize= fontsize_word)
        ax011.set_title(model+"_p moving average "+str(latitude)+"$\degree$\n", fontsize= 10)
        ax011.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax011.set_yticks([])
        ax011.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        ax011a.set_xlim(driftrate_to_velocity(drift)[0], driftrate_to_velocity(drift)[-1])
        ax011a.set_xlabel('Velocity ($km/year$)', fontsize= fontsize_word)

        # cbar = plt.colorbar(MW, aspect = 11.5)
        cbar = plt.colorbar(MW)
        cbar.set_label('Drift signal power ($10^{4}$)', fontsize = fontsize_word)
        cbar.ax.tick_params(labelsize = fontsize_tick, rotation = 90)


    elif model == 'LSMOD.2':
        mov_wind_ka_cent = np.abs(mov_wind_years_cent-2000)/1000
        
        fig  = plt.figure(figsize=(14, 3))    # width, height 14, 3
        grid = fig.add_gridspec(nrows = 1 , ncols= 2, width_ratios = (1, 10),
                                left = 0.1, right = 0.9, bottom = 0.1, top = 0.9,
                                wspace = 0.18)


        # Dipole tilt and axial dipole moment
        grid00 = grid[0].subgridspec(nrows=1, ncols=1)

        color1, color2 = 'black' , 'darkred'
        ax000 = fig.add_subplot(grid00[0])         # dipole tilt
        ax001 = ax000.twiny()                      # axial dipole moment

        ax000.plot(calculate_dipole_tilt(GC_Dict), GC_Dict['ka'], color1)
        ax001.plot(calculate_ADM(GC_Dict)/(1e21), GC_Dict['ka'], color2)

        ax000.set_xlabel('Dipole\ntilt ($\degree$)', fontsize = fontsize_word, color= color1)
        ax000.set_ylabel('Age (ka)', fontsize = fontsize_word)
        ax000.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax000.set_yticks([35, 40, 45])            # LSMOD.2
        ax000.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax000.tick_params(axis= 'y', labelsize = fontsize_tick)

        ax001.set_xlabel('ADM ($ZAm^2$)', fontsize= fontsize_word,  color=color2)
        ax001.set_xlim(100,0)
        ax001.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax001.tick_params(axis= 'y', labelsize = fontsize_tick)


        # Maximum drift signal and moving window
        grid01 = grid[1].subgridspec(nrows=1, ncols=2, width_ratios = (1, 9.5),
                                wspace = 0.05)

        ax010 = fig.add_subplot(grid01[0,0])        # max drift signal
        ax011 = fig.add_subplot(grid01[0,1])        # moving window (MW)
        ax011a = ax011.twiny()
        ax011.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax011a.tick_params(axis= 'x', labelsize = fontsize_tick)

        ax010.plot(max_signal/(1e4), mov_wind_ka_cent, 'k')
        ax010.set_xlabel('Max signal\npower ($10^{4}$)', fontsize= fontsize_word)
        ax010.set_ylabel('MA centered age', fontsize= fontsize_word)
        ax010.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax010.set_yticks([35, 40, 45])            # LSMOD.2
        ax010.tick_params(axis= 'x', labelsize = fontsize_tick)
        ax010.tick_params(axis= 'y', labelsize = fontsize_tick)

        # moving window (MW)
        if latitude == 55:
            vmax_55NorS = 15
            ax010.set_xticks([0, 8])
        if latitude == -55:
            vmax_55NorS = 15
            ax010.set_xticks([0, 10])
        
        # moving window (MW)
        MW = ax011.imshow(moving_window/(1e4), aspect='auto', cmap= cm.lajolla, \
                          vmax = 0, vmin = vmax_55NorS, \
                          extent= [drift[0], drift[-1], mov_wind_ka_cent[-1], mov_wind_ka_cent[0]])
    
        print('Max drift moving average signal:', np.max(moving_window))

        ax011.set_xlabel('WW   Drift rate ($\degree /year$)   EW', fontsize= fontsize_word)
        ax011.set_title(model+"_p moving average "+str(latitude)+"$\degree$\n", fontsize= 10)
        ax011.set_ylim(mov_wind_ka_cent[-1], mov_wind_ka_cent[0])
        ax011.set_yticks([])
        ax011.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        ax011a.set_xlim(driftrate_to_velocity(drift)[0], driftrate_to_velocity(drift)[-1])
        ax011a.set_xlabel('Velocity ($km/year$)', fontsize= fontsize_word)

        # cbar = plt.colorbar(MW, aspect = 3.5)
        cbar = plt.colorbar(MW)
        cbar.set_label('Drift signal power ($10^{4}$)', fontsize = fontsize_word)
        cbar.ax.tick_params(labelsize = fontsize_tick, rotation = 90)

    # ax000.tick_params(width=0.25)
    # ax010.tick_params(width=0.25)
    # ax001.tick_params(width=0.25)
    ax011.tick_params(width=0.25)
    ax011a.tick_params(width=0.25)


    for axis in ['top','bottom','left','right']:
        # ax000.spines[axis].set_linewidth(0.2)
        # ax010.spines[axis].set_linewidth(0.2)
        # ax001.spines[axis].set_linewidth(0.2)
        ax011.spines[axis].set_linewidth(0.2)
        ax011a.spines[axis].set_linewidth(0.2)


if __name__ == '__main__':
    main()


    ##### Code for the radon methods of a timelongitude plot

    # drift               = 0.103
    # theta_degrees       = np.rad2deg( np.arctan(360/(drift*x_pixel*dt)) )
    # print(90-theta_degrees)

    # drift_low, drift_high, drift_inc           = 0.01, 0.3, 0.005

    # projection_degrees = drift_to_projection(drift_low, drift_high, drift_inc, x_pixel, dt)

    # print(projection_degrees)

    # print(projection_degrees[39])

    # # timelongitude[0, 0] = 1e-4
    # # timelongitude[-1, -1] = 1e-4


    # radon_image = skimage.transform.radon(timelongitude, theta = projection_degrees, circle = False, preserve_range = True)

    # plt.figure()
    # plt.imshow(timelongitude)

    # plt.figure()
    # plt.imshow(radon_image)


    # plt.figure(figsize= (5, 3))
    # plt.plot(radon_image[:, 39], color = 'k')
    # # plt.plot(radon_image[:, 39])

    # print(radon_image.shape)

    
    # year_start_index        = np.abs(GC_Dict['year']-year_start).argmin()
    # year_end_index          = np.abs(GC_Dict['year']-year_end).argmin()
    # years                   = GC_Dict['year'][year_start_index:year_end_index] 
    # kas                     = GC_Dict['ka'][year_start_index:year_end_index]

    # time_longitude.plot_timelongitude(timelongitude, model, years, kas , latitude, milliTesla, high_pass, axisymm_state, date)
    # time_longitude.save_timelongitude_fig(high_pass, axisymm_state)


    # plt.figure()
    # plt.plot(np.arange( -180, 180, 2 ), timelongitude.sum(axis=0), label = 'Gaussian Noise')
    
    # # timelongitude_model   = np.loadtxt('/Users/nclizzie/Documents/Research/Time_Longitude/GGF100k/GGF100k_TL_allcoeffs.txt')
    # timelongitude_model   = np.loadtxt('/Users/nclizzie/Documents/Research/Time_Longitude/GGFSS70/GGFSS70_TL_nonaxisym.txt')
    # plt.plot(np.arange( -180, 180, 2 ), (timelongitude_model/(1e6)).sum(axis=0), label = 'GGFSS70')
    # plt.xlabel('Longitude ($\degree$)', fontsize = 15)
    # plt.ylabel('mT', fontsize = 15)
    # plt.legend()
    # plt.xlim(-180, 180)

    # plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/longitude_sum_TL.pdf', format="pdf", bbox_inches="tight")