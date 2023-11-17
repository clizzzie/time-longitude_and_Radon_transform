import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from sklearn import linear_model
import pyshtools as pysh

# Personal modules and functions:
# Module used to create a dictionary for the time-varying Gauss coefficients
import array_to_dictionary_mod
# Lambda function used to calculate the total number of Gauss coefficients
numOfGCLambda       = lambda l: int(np.sum(2*np.arange(1, l+1)+1))


def main(model_faux, axisymm_state_faux, high_pass_faux, year_start_faux, year_end_faux, latitude_faux = 55 , freq_cut_faux = 4000, date_faux = "19_October_2023"):
    """
    Creates the input document for magmap
    Call magmap to calculate the magnetic field components
    has a for loop to calculate all the timelongitude plot
    """

    global model
    global degree
    global axisymm_state
    global high_pass
    global year_start, year_end
    global longitude_start, longitude_end, longitude_spacing
    global latitude
    global r_surface, r_CMB
    global milliTesla
    global date
    global freq_cut

    model           = model_faux
    axisymm_state   = axisymm_state_faux
    high_pass       = high_pass_faux
    year_start      = year_start_faux
    year_end        = year_end_faux
    latitude        = latitude_faux
    date            = date_faux
    freq_cut        = freq_cut_faux

    assert (high_pass == "HP_on") or (high_pass == "HP_off")
    assert (axisymm_state =="axi_keep") or (axisymm_state == "axi_remove_TA") or (axisymm_state == "axi_remove_lin_reg")

    longitude_start     = -180
    longitude_end       = 180
    longitude_spacing   = 2             # degree spacing for long0itude
    r_surface           = 6371e3        # radius of the Earth's surface in meters
    r_CMB               = 3485e3        # radius of the CMB in meters
    milliTesla          = "mT_on"       # Converts nT to mT

    if model == "GGFSS70":
        degree          = 6            # Gauss coefficient degree and order
    elif model == "GGFMB":
        degree          = 6
    else:
        degree          = 10            # Gauss coefficient degree and order
    

    # Loads the Gauss Coefficient Dictionary from the appropriate directory
    GC_Dict = load_GC_Dict(model, high_pass)

    # Finds the closest time index for the year
    year_start_index        = np.abs(GC_Dict['year']-year_start).argmin()
    year_end_index          = np.abs(GC_Dict['year']-year_end).argmin()
    years                   = GC_Dict['year'][year_start_index:year_end_index]
    kas                     = GC_Dict['ka'][year_start_index:year_end_index]
    year_spacing            = GC_Dict['year'][1]-GC_Dict['year'][0]
    print('The closest starting year: ', GC_Dict['year'][year_start_index], ' and the closest end year: ', GC_Dict['year'][year_end_index])

    longitude       = np.arange(longitude_start, longitude_end, longitude_spacing)

    GC_magmap_list  = array_to_dictionary_mod.get_GClist(degree, numOfGCLambda(degree))[1:]

    timelongitude   = create_timelongitude(GC_Dict, GC_magmap_list, year_start_index, year_end_index, longitude, axisymm_state, milliTesla)

    print('length of the years: ', len(GC_Dict['year'][year_start_index:year_end_index]))
    print('length of the longitude: ', len(longitude))
    print('The shape of time-longitude is:', timelongitude.shape)

    if model == 'ledt002' and latitude == 60:
        plot_timelongitude_ledt002(timelongitude, model, years, kas , latitude, milliTesla, high_pass, axisymm_state, date)
    if model == 'GGF100k':
        plot_timelongitude_ggf100k(timelongitude, model, years, kas , latitude, milliTesla, high_pass, axisymm_state, date)
    if model == 'GGFSS70':
        plot_timelongitude_ggfss70(timelongitude, model, years, kas , latitude, milliTesla, high_pass, axisymm_state, date)
    if model == 'LSMOD.2':
        plot_timelongitude_lsmod2(timelongitude, model, years, kas , latitude, milliTesla, high_pass, axisymm_state, date)
    if model == 'GGFMB':
        plot_timelongitude_ggfmb(timelongitude, model, years, kas , latitude, milliTesla, high_pass, axisymm_state, date)
    
    
    # TL_folder_mac = '/Users/nclizzie/Documents/Research/Time_Longitude/'
    TL_folder_Etote = '/Volumes/clizzie_Etote/Time_longitude/'

    save_timelongitude_fig(high_pass, axisymm_state)

    save_timelongitude_txt(timelongitude, years, model, high_pass, axisymm_state, milliTesla, TL_folder = TL_folder_Etote)


    years_complete  = GC_Dict['year'][year_start_index:year_end_index]
    x_pixels        = timelongitude.shape[1]
    y_pixels        = timelongitude.shape[0]

    if milliTesla == "mT_on":
        timelongitude = timelongitude*(1e6)
        print('Time-longitiude plot in nT')

    return timelongitude, years_complete, x_pixels, y_pixels

############################################################################### 


def load_GC_Dict(model:str, high_pass:str):
    """ Loads the Gauss Coefficient dictionary
    """

    if      (model   == 'pfm9k.1a')   and (high_pass == 'HP_on'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/PFM9k_ready_to_load/pfm9k1a_coeffs_surface_10years_HPfiltered.txt')
    elif    (model   == 'pfm9k.1a')   and (high_pass == 'HP_off'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/PFM9k_ready_to_load/pfm9k1a_coeffs_surface_10years.txt')
    elif    (model   == 'GaussianNoise') and (axisymm_state == 'axi_keep'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/Gaussian_noise/Gaussian_noise.txt')
    elif    (model   == 'GaussianNoise') and (axisymm_state == 'axi_remove_TA'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/Gaussian_noise/Gaussian_noise_nonaxisymmetric.txt')
    elif    (model   == 'CALS10k.2')  and (high_pass == 'HP_on'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/CALS10k2_ready_to_load/CALS10k2_coeffs_surface_10years_HPfiltered.txt')
    elif    (model   == 'CALS10k.2')  and (high_pass == 'HP_off'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/CALS10k2_ready_to_load/CALS10k2_coeffs_surface_10years.txt')


    elif    (model   == 'GGF100k')    and (high_pass == 'HP_on'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGF100K_ready_to_load/ggf100k_coeffs_surface_10years_'+str(freq_cut)+'HPfiltered.txt')
    elif    (model   == 'GGF100k')    and (high_pass == 'HP_off'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGF100K_ready_to_load/ggf100k_coeffs_surface_10years.txt')


    elif    (model   == 'GGFSS70')    and (high_pass == 'HP_on'):
        # GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGFSS70k_ready_to_load/GGFSS70k_coeffs_surface_10years_4000HPfiltered.txt')
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGFSS70k_ready_to_load/ggfss70_coeffs_surface_10years_4000to1300BPfiltered.txt')
    elif    (model   == 'GGFSS70')    and (high_pass == 'HP_off'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGFSS70k_ready_to_load/GGFSS70k_coeffs_surface_10years.txt')
    
    
    elif    (model   == 'LSMOD.2')    and (high_pass == 'HP_on'):
        # GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/LSMOD.2_ready_to_load/LSMOD.2_coeffs_surface_10years_4000HPfiltered.txt')
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/LSMOD.2_ready_to_load/LSMOD.2_coeffs_surface_10years_4000to1300BPfiltered.txt')
    elif    (model   == 'LSMOD.2')    and (high_pass == 'HP_off'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/LSMOD.2_ready_to_load/LSMOD.2_coeffs_surface_10years.txt')


    elif    model   == 'Monika_Pm10_Ra350'  :
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/NDS_ready_to_load/monica/NDS_coeffs_surface_interpolated_Pm10_Ra350.txt')
    elif model      == 'ledt002'            :
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/NDS_ready_to_load/ledt2,39/NDS_coeffs_interpolated_ledt002.txt')
    elif model      == 'ledt039'            :
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/NDS_ready_to_load/ledt2,39/NDS_coeffs_interpolated_ledt039.txt')
    elif    (model   == 'GGFMB')      and (high_pass == 'HP_off'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGFMB_ready_to_load/GGFMB_coeffs_surface_200years.txt')
        GC_Dict["year"] = np.rint(-GC_Dict["year"]*1000+2000).astype(int)
    elif    (model   == 'GGFMB')      and (high_pass == 'HP_on'):
        GC_Dict     = array_to_dictionary_mod.main('/Users/nclizzie/Documents/Research/GGFMB_ready_to_load/GGFMB_coeffs_surface_200years.txt')
        GC_Dict["year"] = np.rint(-GC_Dict["year"]*1000+2000).astype(int)
    else:
        print('Sayonara Sammy')
        exit()

    if model == "ledt002":
        GC_Dict["ka"] = GC_Dict['year']
    else:
        GC_Dict["ka"] = np.abs(GC_Dict['year']-2000)/1000

    return GC_Dict

def create_long_lat_coordinates(lat, long_start, long_end, spacing):
    """ this function will create the longitude and longitude array and check all the inputs
    Input   : lat is latitude
            : long_start is the longitude starting coordinate, start with most westard point
            : long_end is the longitude ending coordinate, end with most eastward point
            : spacing is in degrees
    """

    assert type(lat) == int or float,                           "latitude is not an integer or float"
    assert (lat > -90) and (lat < 90),                          "latitude is not within bounds of 90 degrees north or 90 degrees south"
    assert (long_start >= -360) and (long_start <= 360),        "long_start is not within bounds"
    assert (long_end   >= -360) and (long_end   <= 360),        "long_start is not within bounds"
    assert long_start < long_end,                               "longitude start point is greater than longitude ending point"
    assert type(long_start) and type(long_end) == int or float, "longitudes are not integers or floats"

    assert type(spacing) == int or float,                       "spacing is not an integer or float"
    assert spacing > 0,                                         "spacing is not greater than zero"

    # create long (phi) and lati (theta) for a time longitude plot
    longitude = np.arange(long_start, long_end, spacing)
    latitude  = lat*np.ones(len(longitude))

    return longitude, latitude

def create_SH_text(GC_Dict, GC_magmap_list, year , year_start_index, year_end_index, axisymm_state: str, SH_txt = 'SH_coefficients.sh'):
    """Create spherical harmonic text for pysh, just one time increment of spherical harmonics
    Input   : GC_Dict is the Gauss Coefficient dictionary from array_to_dictionary
            : GC_magmap_list is the list of Gauss coefficeints to call GC_Dict in order
            : year is the evaluation year to call from the GC_Dict
    """

    assert type(GC_Dict)        == dict,            "GC_Dict is not a dictionary"
    assert type(GC_magmap_list) == list,            "GC magmap list is not a list"
    assert type(SH_txt)         == str,             "SH_txt is not a string"
    assert type(year)           == int or float,    "year is not an integer or float"
    assert axisymm_state == 'axi_keep' or 'axi_remove_TA' or 'axi_remove_lin_reg'

    year_index = int(np.where(GC_Dict['year'] == year)[0])

    if axisymm_state == 'axi_remove_lin_reg':
        m_Dict     = {}
        
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

        for m_0 in m_equals_zero:
            m_Dict[m_0]  = linear_model.LinearRegression(fit_intercept=True).fit(GC_Dict['year'][year_start_index:year_end_index].reshape((-1,1)), GC_Dict[m_0][year_start_index:year_end_index])


    if axisymm_state == 'axi_remove_TA':        # Takes out the time-averaged axisymmetric parts of the field
        print('Removing time-average of the zonal coefficients for '+str(year))

    elif axisymm_state == 'axi_remove_lin_reg': # Takes out the linear regression of the axisymmetric parts of the field
        print('Removing linear regression of the zonal coefficients for '+str(year))

    elif axisymm_state == 'axi_keep':
        print('Keeping the zonal coefficients for '+str(year))
 


    with open(SH_txt, mode='w+') as f:
        f.write('# l m g_l^m (cosine) h_l^m (sine) \n')

        for index in range(len(GC_magmap_list)):    # 0 to len of GC_magmap_list 
            index_ = GC_magmap_list[index].find('_', 1)
            l = GC_magmap_list[index][1:index_]
            m = GC_magmap_list[index][index_+1:]
        
            if ((axisymm_state == 'axi_remove_TA') and (int(m) == 0)):        # Takes out the time-averaged axisymmetric parts of the field
                # print('Removing time-average of ', GC_magmap_list[index])
                m_equal_zero = 'g'+str(l)+'_'+str(m)
                assert m_equal_zero == GC_magmap_list[index]
                cosine_coeff = GC_Dict[GC_magmap_list[index]][year_index]
                f.write(str(l)+ ' ' + str(m) + ' ' + str(cosine_coeff - np.mean(GC_Dict[m_equal_zero][year_start_index:year_end_index])) + ' ' + str(0) + '\n')

            elif ((axisymm_state == 'axi_remove_lin_reg') and (int(m) == 0)): # Takes out the linear regression of the axisymmetric parts of the field
                # print('Removing linear regression of ', GC_magmap_list[index])
                m_equal_zero = 'g'+str(l)+'_'+str(m)
                assert m_equal_zero == GC_magmap_list[index]
                cosine_coeff = GC_Dict[GC_magmap_list[index]][year_index]

                m_predict_year = m_Dict[m_equal_zero].predict(GC_Dict['year'][year_index].reshape((-1,1)))[0]
                # print(str(cosine_coeff), str(m_predict_year), str(cosine_coeff - m_predict_year))

                f.write(str(l)+ ' ' + str(m) + ' ' + str(cosine_coeff - m_predict_year) + ' ' + str(0) + '\n')

            elif ((GC_magmap_list[index][0] == 'g') and (int(m) == 0) and axisymm_state == 'axi_keep'):
                # print('Not removing time-average of ', GC_magmap_list[index])
                cosine_coeff = GC_Dict[GC_magmap_list[index]][year_index]
                f.write(str(l)+ ' ' + str(m) + ' ' + str(cosine_coeff) +' '+ str(0)+ '\n')

            elif (GC_magmap_list[index][0] == 'h'):
                # print('continuing onto because of h')
                continue

            else:
                # print('g and h')
                cosine_coeff = GC_Dict[GC_magmap_list[index]][year_index]
                sine_coeff   = GC_Dict[GC_magmap_list[index+1]][year_index]
                f.write(str(l)+ ' ' + str(m) + ' ' + str(cosine_coeff) +' '+ str(sine_coeff)+ '\n')

                assert GC_magmap_list[index][1:] == GC_magmap_list[index+1][1:], 'no matching g and h coefficients'


def create_timelongitude(GC_Dict:dict, GC_magmap_list, year_start_index, year_end_index, longitude, axisymm_state:str, milliTesla:str):
    """ Creates the timelongitude matrix by calculating B_r in a time for loop
    """

    assert GC_Dict['year'][year_start_index] < GC_Dict['year'][year_end_index],   'year_start is greater than year_end'

    years           = GC_Dict['year'][year_start_index:year_end_index]
    long, lati      = create_long_lat_coordinates(latitude, longitude_start, longitude_end, longitude_spacing)

    timelongitude   = np.zeros( (len(years), len(longitude)) )
    index = 0
    for year in years:

        create_SH_text(GC_Dict, GC_magmap_list, year, year_start_index, year_end_index, axisymm_state)
        class_model     = pysh.SHMagCoeffs.from_file('/Users/nclizzie/Documents/Research/SH_coefficients.sh', format= 'shtools', r0=r_surface, header= False, lmax=degree)
        B_r             = calculate_Br_at_CMB(class_model = class_model, long_1D=long, lati_1D=lati )

        timelongitude[index, :]   = B_r

        index+=1
        print(year)

    if milliTesla == "mT_on":
        timelongitude = timelongitude/(1e6)
        print('Time-longitude converted from nT to mT')

    return timelongitude

def calculate_Br_at_CMB(class_model, long_1D:np.ndarray, lati_1D:np.ndarray):
    """Calculates B_r at the CMB
    """
    
    assert long_1D.shape == lati_1D.shape

    # Downward continue to the CMB
    expanded_model  = class_model.change_ref(r_CMB)
    # Calculate B at the CMB
    expanded_model1 = expanded_model.expand(lon = long_1D, lat = lati_1D)

    # Extract B_r from the class
    B_r = np.zeros(len(expanded_model1))
    for i in range(len(expanded_model1)):
        B_r[i] = expanded_model1[i][0]

    assert B_r.shape == long_1D.shape

    return B_r


def save_timelongitude_txt(timelongitude, years, model, high_pass, axisymm_state, milliTesla, TL_folder = '/Users/nclizzie/Documents/Research/Time_Longitude/'):
    """ Saves the time-longitude as a text document
    """
    assert type(timelongitude) and type(years)                              == np.ndarray   , 'time-longitude or years are not np.ndarray'
    assert type(model)                                                      == str          , 'date or model is not a string'
    assert type(high_pass) and type(axisymm_state) and type(milliTesla)     == str          , 'high_pass, axisymmetric, or milliTesla is not a str'
    assert timelongitude.shape[0] == years.shape[0]                                         , 'time-longitude and years are not equal in size'


    if      (high_pass == "HP_on") and (axisymm_state == "axi_remove_TA"):
        np.savetxt(TL_folder+model+'/'+model+'_years_for_TL_HP'+str(freq_cut)+'_Axi_TA_removed.txt', years)
        if milliTesla == "mT_on":
            np.savetxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_Axi_TA_removed.txt', timelongitude*1e6)
        else:
            np.savetxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_Axi_TA_removed.txt', timelongitude)
    
    elif    (high_pass == "HP_on") and (axisymm_state == "axi_remove_lin_reg"):
        np.savetxt(TL_folder+model+'/'+model+'_years_for_TL_HP'+str(freq_cut)+'_Axi_removed_byLR.txt', years)
        if milliTesla == "mT_on":
            np.savetxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_Axi_removed_byLR.txt', timelongitude*1e6)
        else:
            np.savetxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_Axi_removed_byLR.txt', timelongitude)

    elif    (high_pass == "HP_on") and (axisymm_state == "axi_keep"):
        np.savetxt(TL_folder+model+'/'+model+'_years_for_TL_HP'+str(freq_cut)+'_allcoeffs.txt', years)
        if milliTesla == "mT_on":
            np.savetxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_allcoeffs.txt', timelongitude*1e6)
        else:
            np.savetxt(TL_folder+model+'/'+model+'_TL_HP'+str(freq_cut)+'_allcoeffs.txt', timelongitude)

    elif    (high_pass == "HP_off") and (axisymm_state == "axi_remove_TA"):
        np.savetxt(TL_folder+model+'/'+model+'_years_for_TL_Axi_TA_removed.txt', years)
        if milliTesla == "mT_on":
            np.savetxt(TL_folder+model+'/'+model+'_TL_Axi_TA_removed.txt', timelongitude*1e6)
        else:
            np.savetxt(TL_folder+model+'/'+model+'_TL_Axi_TA_removed.txt', timelongitude)

    elif    (high_pass == "HP_off") and (axisymm_state == "axi_remove_lin_reg"):
        np.savetxt(TL_folder+model+'/'+model+'_years_for_TL_Axi_removed_byLR.txt', years)
        if milliTesla == "mT_on":
            np.savetxt(TL_folder+model+'/'+model+'_TL_Axi_removed_byLR.txt', timelongitude*1e6)
        else:
            np.savetxt(TL_folder+model+'/'+model+'_TL_Axi_removed_byLR.txt', timelongitude)

    elif    (high_pass == "HP_off") and (axisymm_state == "axi_keep"):
        np.savetxt(TL_folder+model+'/'+model+'_years_for_TL_allcoeffs.txt', years)
        if milliTesla == "mT_on":
            np.savetxt(TL_folder+model+'/'+model+'_TL_allcoeffs.txt', timelongitude*1e6)
        else:
            np.savetxt(TL_folder+model+'/'+model+'_TL_allcoeffs.txt', timelongitude)
    else:
        print('Sayonara Sammy')

    print('All time-longitude documents are saved in nT')

def save_timelongitude_fig(high_pass, axisymm_state):
    """ Saves the time-longitude figure
    """

    assert type(date) and type(model)                                       == str          , 'date or model is not a string'
    assert type(high_pass) and type(axisymm_state)                          == str          , 'high_pass, axisymmetric, or milliTesla is not a str'


    if      high_pass == "HP_on" and axisymm_state == "axi_remove_TA":
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_HP'+str(freq_cut)+'_Axi_TA_removed.pdf', format="pdf", bbox_inches="tight")

    if      high_pass == "HP_on" and axisymm_state == "axi_remove_lin_reg":
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_HP'+str(freq_cut)+'_Axi_removed_byLR.pdf', format="pdf", bbox_inches="tight")

    elif    high_pass == "HP_on" and axisymm_state == "axi_keep":
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_HP'+str(freq_cut)+'_allcoeff.pdf', format="pdf", bbox_inches="tight")

    elif    high_pass == "HP_off" and axisymm_state == "axi_remove_TA":
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_Axi_TA_removed.pdf', format="pdf", bbox_inches="tight")

    elif    high_pass == "HP_off" and axisymm_state == "axi_remove_lin_reg":
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_Axi_removed_byLR.pdf', format="pdf", bbox_inches="tight")

    elif    high_pass == "HP_off" and axisymm_state == "axi_keep":
        plt.savefig('/Users/nclizzie/Documents/Research/Meeting_notes/'+date+'/'+model+'_allcoeff.pdf', format="pdf", bbox_inches="tight")

    else:
        print('Sayonara Sammy')

def plot_timelongitude(timelongitude: np.ndarray, model: str, years: np.ndarray, kas: np.ndarray , latitude, milliTesla, high_pass, axisymm_state, date):
    """Plots the time-longitude
    """

    if years[0] < years[-1]:
        print('The years were reverse! For plotting only the time-longitude was flipped.')
        timelongitude   = np.flipud(timelongitude)
        years           = np.flipud(years)
        kas             = np.flipud(kas)

    # fig  = plt.figure(figsize=(4,4)) 
    # fig  = plt.figure(figsize=(4,4.5))   # 5,8    # 4,5.5
    fig  = plt.figure(figsize=(4.5,9.15))   # 5,8    # 4,5.5  
    grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                        left = 0.1, right = 0.9, bottom = 0.1, top = 0.9
                        )
    ax = fig.add_subplot(grid[0])

    TL = ax.imshow(timelongitude, aspect='auto', cmap = cm.vik , \
        vmin = -np.max(np.abs(timelongitude)) ,vmax= np.max(np.abs(timelongitude)), \
        # vmin = -0.55 ,vmax= 0.55, \       # for 50-30 ka
        # vmin = -0.23 ,vmax= 0.23, \
        extent= [longitude_start, longitude_end, kas[-1], kas[0]] )
    
    print('vmin: ', -np.max(np.abs(timelongitude)), 'vmax: ', np.max(np.abs(timelongitude)))

    star1_index = np.abs(kas-798).argmin()
    star2_index = np.abs(kas-794.2).argmin()
    star3_index = np.abs(kas-783).argmin()

    ax.plot(0, 798, '*', color = 'dimgray', markersize = 10)
    ax.plot(0, 794.2, '*', color = 'dimgray', markersize = 10)
    ax.plot(0, 783, '*', color = 'dimgray', markersize = 10)


    # ax.axhline(41, color = 'gray', linestyle = '--')    # linestyle = '-'
    # plt.plot([100, -180], [0.75, 0.77], ls="--", c=".4")  
    # plt.plot([180, -180], [0.76, 0.7744], ls="--", c=".4")

    # ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
    # ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)

    if model == "ledt002":
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
        # # # Westward
        # # plt.plot([180, -180], [1.0301, 1.0408],  ls="--", dashes=(5, 20), c=color1, linewidth = 0.72)
        # # # plt.plot([180, -180], [1.0261, 1.0368], ls="--", c=".4", linewidth = 0.5)
        # # # plt.plot([180, -180], [1.0135, 1.0255], ls="--", c=".4", linewidth = 0.5)
        # # plt.plot([180, -180], [1.0195, 1.0302],  ls="--", dashes=(5, 20), c=color1, linewidth = 0.72)

        # # plt.plot([180, -180], [0.9663, 0.9772] ,ls="--", dashes=(5, 20),  c=color1, linewidth = 0.72)
        # # plt.plot([180, -180], [0.9439, 0.9558] ,ls="--", dashes=(5, 20),  c=color1, linewidth = 0.72)
        # # plt.plot([180, -180], [0.9379, 0.9488],ls="--", dashes=(5, 20),  c=color1, linewidth = 0.72)

        # # Eastward
        # plt.plot([-180, -19.7], [0.9082, 0.913] , ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)
        # plt.plot([180, -180], [0.9111, 0.8982], ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)
        # plt.plot([180, -180], [0.8959, 0.8830], ls="--", dashes=(5, 20), c=color2, linewidth = 0.72)
        # plt.plot([180, -62.3], [0.8783, 0.8710], ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)
        # plt.plot([180, -62.3], [0.8654, 0.8581], ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)

        # # # Westward
        # # plt.plot([180, -180], [0.8223, 0.8367], ls="--", dashes=(5, 20),  c=color3, linewidth = 0.72)
        # # plt.plot([180, -180], [0.8099, 0.8243], ls="--", dashes=(5, 20),  c=color3, linewidth = 0.72)
        # # plt.plot([180, -180], [0.7899, 0.8099], ls="--", dashes=(5, 20),  c=color3, linewidth = 0.72)
    
    # if model == "GGF100k" and latitude == 555:
    #     color1, color2, color3 = "royalblue" , "teal", "deeppink"
    #     # Westward
    #     ax.plot([-120, 180], [30.1, 33.7],  ls="--", c=color3, linewidth = 0.72)   # 0.083
    #     ax.plot([-180, 180], [31.3, 35.6],  ls="--", c=color3, linewidth = 0.72)   # 0.068
    #     ax.plot([-180, 140], [35.7, 39.6],  ls="--", c=color3, linewidth = 0.72)   # 0.0821
    #     # plt.plot([-180, 180], [42.6, 39.8],  ls="--", c=color1, linewidth = 0.72)

    #     # Eastward
    #     ax.plot([-66, 140], [41.6, 39.6],  ls="--", c=color1, linewidth = 0.72)    # 0.103
    #     ax.plot([-180, 180], [45.7, 41.3],  ls="--", c=color1, linewidth = 0.72)   # 0.082
    #     ax.plot([-180, 180], [48.8, 45.8],  ls="--", c=color1, linewidth = 0.72)   # 0.12
    #     ax.plot([-35, 180],  [50, 48.7],    ls="--", c=color1, linewidth = 0.72)   # 0.165

    #     ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)

    if model == "GGF100k" and latitude == -555:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        # Eastward
        ax.plot([-180, 180],  [44, 38],  ls="--", c=color1, linewidth = 0.72)   # 0.06
        ax.plot([-180, 180],  [40.8, 35],    ls="--", c=color1, linewidth = 0.72)   # 0. 62

        ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)

    if model == "GGFSS70" and latitude == 55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
        # # Westward
        # ax.plot([-70, 180], [30.1, 33.1],  ls="--", c=color3, linewidth = 0.72) # 0.083
        # ax.plot([-80, 180], [32.9, 36],  ls="--", c=color3, linewidth = 0.72)   # 0.084
        ax.plot([-180, 70], [39, 40.36],  ls="--", c=color3, linewidth = 0.72)  # 0.18

        # # Eastward
        # ax.plot([-180, 70], [42.6, 40.36],  ls="--", c=color1, linewidth = 0.72)    #0.11

        # # Westward
        # ax.plot([-60, 180], [40.6, 43.1],  ls="--", c=color3, linewidth = 0.72) # 0.096
        # ax.plot([-180, 32], [45, 47.7],  ls="--", c=color3, linewidth = 0.72)   # 0.078
        
        ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
        ax.axvspan(90, 180 , facecolor= 'gray', alpha = 0.2)
        

    if model == "GGFSS70" and latitude == -55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        # Eastward
        ax.plot([-108, 180], [35.25, 32],  ls="--", c=color1, linewidth = 0.72)    # 0.088
        ax.plot([-180, 180], [38, 34],  ls="--", c=color1, linewidth = 0.72)    # 0.09
        ax.plot([-43, 180], [39.5, 37],  ls="--", c=color1, linewidth = 0.72)    # 0.089
        
        # Westward
        ax.plot([-180, 180], [38.5, 43],  ls="--", c=color3, linewidth = 0.72)   # 0.08
        ax.plot([-180, 180], [39.5, 44],  ls="--", c=color3, linewidth = 0.72)   # 0.08
        
        # # Eastward
        # ax.plot([-180, 180], [41.5, 39.5],  ls="--", c=color1, linewidth = 0.72)    # 0.18
        # ax.plot([-180, 180], [43.5, 41.5],  ls="--", c=color1, linewidth = 0.72)    # 0.18
        
        ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)
        ax.axvspan(130, 180 , facecolor= 'gray', alpha = 0.2)

    if model == "LSMOD.2" and latitude == 55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
        # Westward
        # ax.plot([-180, 43], [31.3, 34],  ls="--", c=color3, linewidth = 0.72)   # 0.0825
        # ax.plot([-180, 43], [32.3, 34.9],  ls="--", c=color3, linewidth = 0.72)   # 0.0857
        
        # # Eastward
        # ax.plot([-180, -12], [36, 34.3],  ls="--", c=color1, linewidth = 0.72) # 0.11
        # ax.plot([-180, -15], [37, 35],  ls="--", c=color1, linewidth = 0.72)   # 0.0975

        # Westward
        ax.plot([-106, 43], [38.4, 40.2],  ls="--", c=color3, linewidth = 0.72)  # 0.083
        ax.plot([-180, 10], [39.9, 42.5],  ls="--", c=color3, linewidth = 0.72)  # 0.073

        # Eastward
        ax.plot([-180, 43], [42.9, 40.2],  ls="--", c=color1, linewidth = 0.72)    # 0.0826
        ax.plot([-180, 140], [44.7, 40.8],  ls="--", c=color1, linewidth = 0.72)   # 0.082
        
        ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
        

    if model == "LSMOD.2" and latitude == -55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        # # Eastward
        # ax.plot([-108, 80], [35.25, 33.3],  ls="--", c=color1, linewidth = 0.72)    #0.096
        # ax.plot([-96, 60], [40, 38.2],  ls="--", c=color1, linewidth = 0.72)    # 0.087
        # ax.plot([-90, 130], [41.1, 38.6],  ls="--", c=color1, linewidth = 0.72)    # 0.088
    
        # Westward
        ax.plot([-96, 80], [40, 42],  ls="--", c=color3, linewidth = 0.72)      # 0.088
        ax.plot([-90, 65], [41.1, 43],  ls="--", c=color3, linewidth = 0.72)    # 0.0815
        # ax.plot([-180, 180], [43.5, 48.5],  ls="--", c=color3, linewidth = 0.72)    # 0.072

        ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)
        ax.axvspan(130, 180 , facecolor= 'gray', alpha = 0.2)

    # ax.set_xticks([])
    # ax.set_yticks([])

    ax.set_xticks([-180, -90, 0, 90, 180])
    # ax.set_yticks([50, 45, 40, 35, 30])
    # ax.set_yticks([80, 82, 84, 86, 88, 90])

    ax.tick_params(axis='x', labelsize = 10)        # labelsize = 13 for models
    ax.tick_params(axis='y', labelsize = 10)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
    #     ax000.spines[axis].set_linewidth(0.2)
    #     ax010.spines[axis].set_linewidth(0.2)
    #     ax001.spines[axis].set_linewidth(0.2)
          ax.spines[axis].set_linewidth(0.15)
    #     ax011a.spines[axis].set_linewidth(0.2)

    ax.set_xlabel('Longitude ($^{\circ}$)', fontsize= 20)
    if model == "ledt002":
        ax.set_ylabel('Diffusion time ($t_d$)', fontsize = 20)
    else:
        ax.set_ylabel('Age (ka)', fontsize = 20)
    # plt.title('Time Longitude for '+ model + ' latitude: ' + str(latitude), fontsize = 15)
    ax.set_title(model + ' latitude: ' + str(latitude)+'\n', fontsize = 10)

    cbar = plt.colorbar(TL)

    if milliTesla == "mT_on":
        # plt.colorbar(label = '$B_r$ (mT)', fontsize = 15)
        cbar.set_label("$B_r$ (mT)",  fontsize = 15)

    else:
        cbar.set_label("$B_r$",  fontsize = 15)

    # cbar.ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    # cbar.ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # cbar.ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    cbar.ax.tick_params(labelsize=10)                # labelsize = 13 for models


def plot_timelongitude_ggfmb(timelongitude: np.ndarray, model: str, years: np.ndarray, kas: np.ndarray , latitude, milliTesla, high_pass, axisymm_state, date):
    """Plots the time-longitude
    """

    if years[0] < years[-1]:
        print('The years were reverse! For plotting only the time-longitude was flipped.')
        timelongitude   = np.flipud(timelongitude)
        years           = np.flipud(years)
        kas             = np.flipud(kas)

    # fig  = plt.figure(figsize=(4,4)) 
    # fig  = plt.figure(figsize=(4,4.5))   # 5,8    # 4,5.5
    fig  = plt.figure(figsize=(4.5,9.15))   # 5,8    # 4,5.5  
    grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                        left = 0.1, right = 0.9, bottom = 0.1, top = 0.9
                        )
    ax = fig.add_subplot(grid[0])

    TL = ax.imshow(timelongitude, aspect='auto', cmap = cm.vik , \
        vmin = -np.max(np.abs(timelongitude)) ,vmax= np.max(np.abs(timelongitude)), \
        # vmin = -0.55 ,vmax= 0.55, \       # for 50-30 ka
        # vmin = -0.23 ,vmax= 0.23, \
        extent= [longitude_start, longitude_end, kas[-1], kas[0]] )
    
    print('vmin: ', -np.max(np.abs(timelongitude)), 'vmax: ', np.max(np.abs(timelongitude)))


    # ax.set_xticks([])
    # ax.set_yticks([])

    ax.set_xticks([-180, -90, 0, 90, 180])
    # ax.set_yticks([50, 45, 40, 35, 30])
    # ax.set_yticks([80, 82, 84, 86, 88, 90])

    ax.tick_params(axis='x', labelsize = 10)        # labelsize = 13 for models
    ax.tick_params(axis='y', labelsize = 10)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
    #     ax000.spines[axis].set_linewidth(0.2)
    #     ax010.spines[axis].set_linewidth(0.2)
    #     ax001.spines[axis].set_linewidth(0.2)
          ax.spines[axis].set_linewidth(0.15)
    #     ax011a.spines[axis].set_linewidth(0.2)

    ax.set_xlabel('Longitude ($^{\circ}$)', fontsize= 20)
    if model == "ledt002":
        ax.set_ylabel('Diffusion time ($t_d$)', fontsize = 20)
    else:
        ax.set_ylabel('Age (ka)', fontsize = 20)
    # plt.title('Time Longitude for '+ model + ' latitude: ' + str(latitude), fontsize = 15)
    ax.set_title(model + ' latitude: ' + str(latitude)+'\n', fontsize = 10)

    cbar = plt.colorbar(TL)

    if milliTesla == "mT_on":
        # plt.colorbar(label = '$B_r$ (mT)', fontsize = 15)
        cbar.set_label("$B_r$ (mT)",  fontsize = 15)

    else:
        cbar.set_label("$B_r$",  fontsize = 15)

    # cbar.ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    # cbar.ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # cbar.ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    cbar.ax.tick_params(labelsize=10)                # labelsize = 13 for models


def plot_timelongitude_lsmod2(timelongitude: np.ndarray, model: str, years: np.ndarray, kas: np.ndarray , latitude, milliTesla, high_pass, axisymm_state, date):
    """Plots the time-longitude
    """

    if years[0] < years[-1]:
        print('The years were reverse! For plotting only the time-longitude was flipped.')
        timelongitude   = np.flipud(timelongitude)
        years           = np.flipud(years)
        kas             = np.flipud(kas)

    fig  = plt.figure(figsize=(4,8)) 
    # fig  = plt.figure(figsize=(4,4.5))   # 5,8    # 4,5.5
    # fig  = plt.figure(figsize=(4.5,9.15))   # 5,8    # 4,5.5  
    grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                        left = 0.1, right = 0.9, bottom = 0.1, top = 0.9
                        )
    ax = fig.add_subplot(grid[0])
    axa = ax.twiny()

    TL = ax.imshow(timelongitude, aspect='auto', cmap = cm.vik , \
        # vmin = -np.max(np.abs(timelongitude)) ,vmax= np.max(np.abs(timelongitude)), \
        vmin = -0.55 ,vmax= 0.55,       # for 50-30 ka
        # vmin = -0.23 ,vmax= 0.23, \
        extent= [longitude_start, longitude_end, kas[-1], kas[0]] )
    
    print('vmin: ', -np.max(np.abs(timelongitude)), 'vmax: ', np.max(np.abs(timelongitude)))

    ax.axhline(41, color = 'gray', linestyle = '--')    # linestyle = '-'

    # ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
    # ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)


    if latitude == 55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
        # Westward
        # ax.plot([-180, 43], [31.3, 34],  ls="--", c=color3, linewidth = 0.72)   # 0.0825
        # ax.plot([-180, 43], [32.3, 34.9],  ls="--", c=color3, linewidth = 0.72)   # 0.0857
        
        # # Eastward
        # ax.plot([-180, -12], [36, 34.3],  ls="--", c=color1, linewidth = 0.72) # 0.11
        # ax.plot([-180, -15], [37, 35],  ls="--", c=color1, linewidth = 0.72)   # 0.0975

        # Westward
        ax.plot([-106, 43], [38.4, 40.2],  ls="--", c=color3, linewidth = 0.72)  # 0.083
        ax.plot([-180, 10], [39.9, 42.5],  ls="--", c=color3, linewidth = 0.72)  # 0.073

        # Eastward
        ax.plot([-180, 43], [42.9, 40.2],  ls="--", c=color1, linewidth = 0.72)    # 0.0826
        ax.plot([-180, 140], [44.7, 40.8],  ls="--", c=color1, linewidth = 0.72)   # 0.082
        
        # ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
        
        axa.set_xlim(longitude_start, longitude_end)
        LSMOD2_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/LSMOD.2_mean_lat_long.txt')
        LSMOD2_lat      = LSMOD2_lat_long[:,0]
        LSMOD2_long     = LSMOD2_lat_long[:,1]
        lat_index = [index for index,value in enumerate(LSMOD2_lat) if value > 35 and value < 75]
        print(LSMOD2_long[lat_index])

        axa.set_xticks(LSMOD2_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'darkgreen')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'
        

    if latitude == -55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        # # Eastward
        # ax.plot([-108, 80], [35.25, 33.3],  ls="--", c=color1, linewidth = 0.72)    #0.096
        # ax.plot([-96, 60], [40, 38.2],  ls="--", c=color1, linewidth = 0.72)    # 0.087
        # ax.plot([-90, 130], [41.1, 38.6],  ls="--", c=color1, linewidth = 0.72)    # 0.088
    
        # Westward
        ax.plot([-96, 80], [40, 42],  ls="--", c=color3, linewidth = 0.72)      # 0.088
        ax.plot([-90, 65], [41.1, 43],  ls="--", c=color3, linewidth = 0.72)    # 0.0815
        # ax.plot([-180, 180], [43.5, 48.5],  ls="--", c=color3, linewidth = 0.72)    # 0.072

        # ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)
        # ax.axvspan(130, 180 , facecolor= 'gray', alpha = 0.2)

        axa.set_xlim(longitude_start, longitude_end)
        LSMOD2_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/LSMOD.2_mean_lat_long.txt')
        LSMOD2_lat      = LSMOD2_lat_long[:,0]
        LSMOD2_long     = LSMOD2_lat_long[:,1]
        lat_index = [index for index,value in enumerate(LSMOD2_lat) if value < -35 and value > -75]
        print(LSMOD2_long[lat_index])

        axa.set_xticks(LSMOD2_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'darkgreen')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'

    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([50, 45, 40, 35, 30])
    # ax.set_yticks([80, 82, 84, 86, 88, 90])

    ax.tick_params(axis='x', labelsize = 13)        # labelsize = 13 for models
    ax.tick_params(axis='y', labelsize = 13)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(0.15)
          axa.spines[axis].set_linewidth(0.15)

    ax.set_xlabel('Longitude ($^{\circ}$)', fontsize= 20)
    ax.set_ylabel('Age (ka)', fontsize = 20)
    ax.set_title(model + ' latitude: ' + str(latitude)+'\n', fontsize = 10)

    cbar = plt.colorbar(TL)

    if milliTesla == "mT_on":
        cbar.set_label("$B_r$ (mT)",  fontsize = 15)

    # cbar.ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    cbar.ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # cbar.ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    cbar.ax.tick_params(labelsize=13)                # labelsize = 13 for models



def plot_timelongitude_ggfss70(timelongitude: np.ndarray, model: str, years: np.ndarray, kas: np.ndarray , latitude, milliTesla, high_pass, axisymm_state, date):
    """Plots the time-longitude
    """

    if years[0] < years[-1]:
        print('The years were reverse! For plotting only the time-longitude was flipped.')
        timelongitude   = np.flipud(timelongitude)
        years           = np.flipud(years)
        kas             = np.flipud(kas)

    fig  = plt.figure(figsize=(4,8)) 
    # fig  = plt.figure(figsize=(4,4.5))   # 5,8    # 4,5.5
    # fig  = plt.figure(figsize=(4.5,9.15))   # 5,8    # 4,5.5  
    grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                        left = 0.1, right = 0.9, bottom = 0.1, top = 0.9
                        )
    ax = fig.add_subplot(grid[0])
    axa = ax.twiny()

    TL = ax.imshow(timelongitude, aspect='auto', cmap = cm.vik , \
        # vmin = -np.max(np.abs(timelongitude)) ,vmax= np.max(np.abs(timelongitude)), \
        vmin = -0.55 ,vmax= 0.55,       # for 50-30 ka
        extent= [longitude_start, longitude_end, kas[-1], kas[0]] )
    
    print('vmin: ', -np.max(np.abs(timelongitude)), 'vmax: ', np.max(np.abs(timelongitude)))

    ax.axhline(41, color = 'gray', linestyle = '--')    # linestyle = '-'

    # ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
    # ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)


    if latitude == 55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
        # # Westward
        # ax.plot([-70, 180], [30.1, 33.1],  ls="--", c=color3, linewidth = 0.72) # 0.083
        # ax.plot([-80, 180], [32.9, 36],  ls="--", c=color3, linewidth = 0.72)   # 0.084
        ax.plot([-180, 70], [39, 40.36],  ls="--", c=color3, linewidth = 0.72)  # 0.183.8

        # # Eastward
        # ax.plot([-180, 70], [42.6, 40.36],  ls="--", c=color1, linewidth = 0.72)    #0.11

        # # Westward
        # ax.plot([-60, 180], [40.6, 43.1],  ls="--", c=color3, linewidth = 0.72) # 0.096
        # ax.plot([-180, 32], [45, 47.7],  ls="--", c=color3, linewidth = 0.72)   # 0.078
        
        # ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
        # ax.axvspan(90, 180 , facecolor= 'gray', alpha = 0.2)


        axa.set_xlim(longitude_start, longitude_end)
        GGFSS70_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGFSS70_lat_long.txt')
        GGFSS70_lat      = GGFSS70_lat_long[:,0]
        GGFSS70_long     = GGFSS70_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGFSS70_lat) if value < 75 and value > 35 ]
        print(GGFSS70_long[lat_index])

        axa.set_xticks(GGFSS70_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'indianred')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'
        

    if latitude == -55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        # Eastward
        ax.plot([-108, 180], [35.25, 32],  ls="--", c=color1, linewidth = 0.72)    # 0.088
        ax.plot([-180, 180], [38, 34],  ls="--", c=color1, linewidth = 0.72)    # 0.09
        ax.plot([-43, 180], [39.5, 37],  ls="--", c=color1, linewidth = 0.72)    # 0.089
        
        # Westward
        ax.plot([-180, 180], [38.5, 43],  ls="--", c=color3, linewidth = 0.72)   # 0.08
        ax.plot([-180, 180], [39.5, 44],  ls="--", c=color3, linewidth = 0.72)   # 0.08
        
        # # Eastward
        # ax.plot([-180, 180], [41.5, 39.5],  ls="--", c=color1, linewidth = 0.72)    # 0.18
        # ax.plot([-180, 180], [43.5, 41.5],  ls="--", c=color1, linewidth = 0.72)    # 0.18
        
        # ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)
        # ax.axvspan(130, 180 , facecolor= 'gray', alpha = 0.2)

        axa.set_xlim(longitude_start, longitude_end)
        GGFSS70_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGFSS70_lat_long.txt')
        GGFSS70_lat      = GGFSS70_lat_long[:,0]
        GGFSS70_long     = GGFSS70_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGFSS70_lat) if value < -35 and value > -75]
        print(GGFSS70_long[lat_index])

        axa.set_xticks(GGFSS70_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'indianred')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'

    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([50, 45, 40, 35, 30])

    ax.tick_params(axis='x', labelsize = 13)        # labelsize = 13 for models
    ax.tick_params(axis='y', labelsize = 13)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(0.15)
          axa.spines[axis].set_linewidth(0.15)

    ax.set_xlabel('Longitude ($^{\circ}$)', fontsize= 20)
    ax.set_ylabel('Age (ka)', fontsize = 20)
    ax.set_title(model + ' latitude: ' + str(latitude)+'\n', fontsize = 10)

    cbar = plt.colorbar(TL)

    if milliTesla == "mT_on":
        cbar.set_label("$B_r$ (mT)",  fontsize = 15)

    cbar.ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    # cbar.ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    cbar.ax.tick_params(labelsize=10)                # labelsize = 13 for models



def plot_timelongitude_ggf100k(timelongitude: np.ndarray, model: str, years: np.ndarray, kas: np.ndarray , latitude, milliTesla, high_pass, axisymm_state, date):
    """Plots the time-longitude
    """

    if years[0] < years[-1]:
        print('The years were reverse! For plotting only the time-longitude was flipped.')
        timelongitude   = np.flipud(timelongitude)
        years           = np.flipud(years)
        kas             = np.flipud(kas)


    fig  = plt.figure(figsize=(4,8))   # 5,8    # 4,5.5  # 4.5, 9.15    # processed 50-30 4,8
    grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                        left = 0.1, right = 0.9, bottom = 0.1, top = 0.9
                        )
    ax = fig.add_subplot(grid[0])
    axa = ax.twiny()
    fontsize_word = 20
    fontsize_tick = 15
    fontsize_small_word = 15

    TL = ax.imshow(timelongitude, aspect='auto', cmap = cm.vik , \
        # vmin = -np.max(np.abs(timelongitude)) ,vmax= np.max(np.abs(timelongitude)), \
        vmin = -0.55 ,vmax= 0.55,       # for 50-30 ka
        extent= [longitude_start, longitude_end, kas[-1], kas[0]] )
    
    print('vmin: ', -np.max(np.abs(timelongitude)), 'vmax: ', np.max(np.abs(timelongitude)))

    ax.axhline(41, color = 'gray', linestyle = '--')    # linestyle = '-'

    # ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
    # ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)

    if latitude == 55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
        # Westward
        ax.plot([-120, 180], [30.1, 33.7],  ls="--", c=color3, linewidth = 0.72)   # 0.083
        ax.plot([-180, 180], [32.3, 37.6],  ls="--", c=color3, linewidth = 0.72)      #0.067
        # ax.plot([-180, 180], [31.3, 35.6],  ls="--", c=color3, linewidth = 0.72)   # 0.068
        ax.plot([-180, 140], [35.7, 39.6],  ls="--", c=color3, linewidth = 0.72)   # 0.0821
        ax.plot([-50, 60], [41, 43.8],  ls="--", c=color3, linewidth = 0.72)   # 0.037
        

        # Eastward
        ax.plot([-66, 140], [41.6, 39.6],  ls="--", c=color1, linewidth = 0.72)    # 0.103
        ax.plot([-180, 180], [45.7, 41.3],  ls="--", c=color1, linewidth = 0.72)   # 0.082
        ax.plot([-180, 180], [48.8, 45.8],  ls="--", c=color1, linewidth = 0.72)   # 0.12
        ax.plot([-35, 180],  [50, 48.7],    ls="--", c=color1, linewidth = 0.72)   # 0.165

        # ax.axvspan(-180, -160 , facecolor= 'gray', alpha = 0.2)
    
        axa.set_xlim(longitude_start, longitude_end)
        GGF100k_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGF100k_lat_long.txt')
        GGF100k_lat      = GGF100k_lat_long[:,0]
        GGF100k_long     = GGF100k_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGF100k_lat) if value < 75 and value > 35 ]
        print(GGF100k_long[lat_index])

        axa.set_xticks(GGF100k_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'dodgerblue')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'


    if latitude == -55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        # Eastward
        # ax.plot([-180, 180],  [41.5, 34.3],    ls="--", c=color1, linewidth = 0.72)   # 0.05
        ax.plot([-180, 75],  [41.5, 36.4],    ls="--", c=color1, linewidth = 0.72)   # 0.05
        ax.plot([-180, 180],  [44.7, 37.3],  ls="--", c=color1, linewidth = 0.72)   # 0.05

        # ax.axvspan(-180, -120 , facecolor= 'gray', alpha = 0.2)
        axa.set_xlim(longitude_start, longitude_end)
        GGF100k_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGF100k_lat_long.txt')
        GGF100k_lat      = GGF100k_lat_long[:,0]
        GGF100k_long     = GGF100k_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGF100k_lat) if value < -35 and value > -75 ]
        print(GGF100k_long[lat_index])

        axa.set_xticks(GGF100k_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'dodgerblue')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'


    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([50, 45, 40, 35, 30])
    # ax.set_yticks([80, 82, 84, 86, 88, 90])

    ax.tick_params(axis='x', labelsize = 13)        # labelsize = 13 for models
    ax.tick_params(axis='y', labelsize = 13)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.15)           # 0.15 for thicker
        axa.spines[axis].set_linewidth(0.15)           # 0.15 for thicker

    ax.set_xlabel('Longitude ($^{\circ}$)', fontsize= 20)
    ax.set_ylabel('Age (ka)', fontsize = 20)
    ax.set_title(model + ' latitude: ' + str(latitude)+'\n', fontsize = 10)

    cbar = plt.colorbar(TL)

    if milliTesla == "mT_on":
        cbar.set_label("$B_r$ (mT)",  fontsize = 15)

    # cbar.ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    cbar.ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.ax.tick_params(labelsize=13)                # labelsize = 13 for models
 

def plot_timelongitude_ggf100k_90to80ka(timelongitude: np.ndarray, model: str, years: np.ndarray, kas: np.ndarray , latitude, milliTesla, high_pass, axisymm_state, date):
    """Plots the time-longitude
    """

    if years[0] < years[-1]:
        print('The years were reverse! For plotting only the time-longitude was flipped.')
        timelongitude   = np.flipud(timelongitude)
        years           = np.flipud(years)
        kas             = np.flipud(kas)


    fig  = plt.figure(figsize=(4,4.5))    # processed 90-80 4, 4.5
    grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                        left = 0.1, right = 0.9, bottom = 0.1, top = 0.9
                        )
    ax = fig.add_subplot(grid[0])
    axa = ax.twiny()
    fontsize_word = 20
    fontsize_tick = 15
    fontsize_small_word = 15

    TL = ax.imshow(timelongitude, aspect='auto', cmap = cm.vik , \
        # vmin = -np.max(np.abs(timelongitude)) ,vmax= np.max(np.abs(timelongitude)), \
        vmin = -0.23 ,vmax= 0.23,       # for 50-30 ka
        extent= [longitude_start, longitude_end, kas[-1], kas[0]] )
    
    print('vmin: ', -np.max(np.abs(timelongitude)), 'vmax: ', np.max(np.abs(timelongitude)))


    if latitude == 55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
    
        axa.set_xlim(longitude_start, longitude_end)
        GGF100k_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGF100k_lat_long.txt')
        GGF100k_lat      = GGF100k_lat_long[:,0]
        GGF100k_long     = GGF100k_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGF100k_lat) if value < 75 and value > 35 ]
        print(GGF100k_long[lat_index])

        axa.set_xticks(GGF100k_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'dodgerblue')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'


    if latitude == -55:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        axa.set_xlim(longitude_start, longitude_end)
        GGF100k_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGF100k_lat_long.txt')
        GGF100k_lat      = GGF100k_lat_long[:,0]
        GGF100k_long     = GGF100k_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGF100k_lat) if value < -35 and value > -75 ]
        print(GGF100k_long[lat_index])

        axa.set_xticks(GGF100k_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'dodgerblue')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'


    if latitude == 20:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"
    
        axa.set_xlim(longitude_start, longitude_end)
        GGF100k_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGF100k_lat_long.txt')
        GGF100k_lat      = GGF100k_lat_long[:,0]
        GGF100k_long     = GGF100k_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGF100k_lat) if value < 40 and value > 0 ]
        print(GGF100k_long[lat_index])

        axa.set_xticks(GGF100k_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'dodgerblue')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'


    if latitude == -20:
        color1, color2, color3 = "royalblue" , "teal", "deeppink"

        axa.set_xlim(longitude_start, longitude_end)
        GGF100k_lat_long = np.loadtxt('/Users/nclizzie/Documents/Paleomagnetic_data/GGF100k_lat_long.txt')
        GGF100k_lat      = GGF100k_lat_long[:,0]
        GGF100k_long     = GGF100k_lat_long[:,1]
        lat_index = [index for index,value in enumerate(GGF100k_lat) if value < 0 and value > -40 ]
        print(GGF100k_long[lat_index])

        axa.set_xticks(GGF100k_long[lat_index])
        axa.tick_params(labeltop = False)
        axa.tick_params(width = 2, color = 'dodgerblue')   # 'dodgerblue' 'darkgreen' 'indianred' 'firebrick'

    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([80, 82, 84, 86, 88, 90])

    ax.tick_params(axis='x', labelsize = 13)        # labelsize = 13 for models
    ax.tick_params(axis='y', labelsize = 13)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.15)           # 0.15 for thicker
        axa.spines[axis].set_linewidth(0.15)           # 0.15 for thicker

    ax.set_xlabel('Longitude ($^{\circ}$)', fontsize= 20)
    ax.set_ylabel('Age (ka)', fontsize = 20)
    ax.set_title(model + ' latitude: ' + str(latitude)+'\n', fontsize = 10)

    cbar = plt.colorbar(TL)

    if milliTesla == "mT_on":
        cbar.set_label("$B_r$ (mT)",  fontsize = 15)

    # cbar.ax.set_yticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cbar.ax.tick_params(labelsize=13)                # labelsize = 13 for models

def plot_timelongitude_ledt002(timelongitude: np.ndarray, model: str, years: np.ndarray, kas: np.ndarray , latitude, milliTesla, high_pass, axisymm_state, date):
    """Plots the time-longitude
    """

    if years[0] < years[-1]:
        print('The years were reverse! For plotting only the time-longitude was flipped.')
        timelongitude   = np.flipud(timelongitude)
        years           = np.flipud(years)
        kas             = np.flipud(kas)


    fig  = plt.figure(figsize=(4.5,9.15))
    grid = fig.add_gridspec(nrows = 1 , ncols= 1,
                        left = 0.1, right = 0.9, bottom = 0.1, top = 0.9
                        )
    ax = fig.add_subplot(grid[0])

    TL = ax.imshow(timelongitude, aspect='auto', cmap = cm.vik , \
        vmin = -np.max(np.abs(timelongitude)) ,vmax= np.max(np.abs(timelongitude)), \
        extent= [longitude_start, longitude_end, kas[-1], kas[0]] )
    
    print('vmin: ', -np.max(np.abs(timelongitude)), 'vmax: ', np.max(np.abs(timelongitude)))

    color1, color2, color3 = "royalblue" , "teal", "deeppink"
    # Westward
    plt.plot([180, -180], [1.0301, 1.0408],  ls="--", dashes=(5, 20), c=color1, linewidth = 0.72)
    # plt.plot([180, -180], [1.0261, 1.0368], ls="--", c=".4", linewidth = 0.5)
    # plt.plot([180, -180], [1.0135, 1.0255], ls="--", c=".4", linewidth = 0.5)
    plt.plot([180, -180], [1.0195, 1.0302],  ls="--", dashes=(5, 20), c=color1, linewidth = 0.72)

    plt.plot([180, -180], [0.9663, 0.9772] ,ls="--", dashes=(5, 20),  c=color1, linewidth = 0.72)
    plt.plot([180, -180], [0.9439, 0.9558] ,ls="--", dashes=(5, 20),  c=color1, linewidth = 0.72)
    plt.plot([180, -180], [0.9379, 0.9488],ls="--", dashes=(5, 20),  c=color1, linewidth = 0.72)

    # Eastward
    plt.plot([-180, -19.7], [0.9082, 0.913] , ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)
    plt.plot([180, -180], [0.9111, 0.8982], ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)
    plt.plot([180, -180], [0.8959, 0.8830], ls="--", dashes=(5, 20), c=color2, linewidth = 0.72)
    plt.plot([180, -62.3], [0.8783, 0.8710], ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)
    plt.plot([180, -62.3], [0.8654, 0.8581], ls="--", dashes=(5, 20),  c=color2, linewidth = 0.72)

    # Westward
    plt.plot([180, -180], [0.8223, 0.8367], ls="--", dashes=(5, 20),  c=color3, linewidth = 0.72)
    plt.plot([180, -180], [0.8099, 0.8243], ls="--", dashes=(5, 20),  c=color3, linewidth = 0.72)
    plt.plot([180, -180], [0.7899, 0.8099], ls="--", dashes=(5, 20),  c=color3, linewidth = 0.72)
    
    ax.set_xticks([-180, -90, 0, 90, 180])

    ax.tick_params(axis='x', labelsize = 12.5)        # labelsize = 13 for models
    ax.tick_params(axis='y', labelsize = 12.5)
    ax.tick_params(width=0.25)
    for axis in ['top','bottom','left','right']:
          ax.spines[axis].set_linewidth(0.15)

    ax.set_xlabel('Longitude ($^{\circ}$)', fontsize= 18)
    ax.set_ylabel('Diffusion time ($t_d$)', fontsize = 18)

    # plt.title('Time Longitude for '+ model + ' latitude: ' + str(latitude), fontsize = 15)
    ax.set_title(model + ' latitude: ' + str(latitude)+'\n', fontsize = 10)

    cbar = plt.colorbar(TL, aspect = 35)

    if milliTesla == "mT_on":
        cbar.set_label("$B_r$ mT_on",  fontsize = 18)
    else:
        cbar.set_label("$B_r$",  fontsize = 18)
    cbar.ax.tick_params(labelsize=12.5)




# def create_magmap_input(magmap_inputfile = 'magmap_input.txt', SH_txt = 'SH_coefficients.txt' , coord_file = 'coordinate_file.txt'):
#     """Create the magmap fortran executable input file
#     Input   : SH_txt is the name of the spherical harmonic text file
#             : coord_file is the geographic coordinate locations where the B_r will be calculated
#     """

#     assert type(SH_txt) and type(coord_file) and type(magmap_inputfile) == str,  'SH_txt or coord_file are not strings'

#     with open(magmap_inputfile, mode = 'w+') as f:
#         f.write('file ' + SH_txt + '\n')
#         f.write('normalize S \n')
#         f.write('site '+ coord_file + '\n')

#         f.write('execute \n')
#         f.write('quit')





if __name__ == '__main__':
    main()
