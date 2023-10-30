import numpy as np

def main(GCtxt):
    """Converts a 2D array to a dictionary with Gauss coefficients as keys.
    Reference 2D_to_dictionary.ipynb for more details on how code works.
    Input   : GCtxt is the 2D array text file.
    Output  : A new dictionary with keys as Gauss coefficients
    """

    assert type(GCtxt)        == str    

    GC_2D                     =  np.loadtxt(GCtxt)
    assert type(GC_2D)        == np.ndarray    , 'GCtxt not an array'

    yearCount, numOfCols      = GC_2D.shape
    totalGC                   = numOfCols-1
    degree                    = get_degree(totalGC)
    GClist                    = get_GClist(degree, totalGC)
    GCdict                    = get_dictionary(GC_2D, GClist)
    print_stats(GCdict)

    return GCdict


########################################################################################

# ### get_degree: Calculates the degree and order
# * Total Gauss Coefficients per year $ =2l +1 $
# * While loop: subtracts from the total coefficients in an increasing order until zero

def get_degree(totalGC):
    """ Calculates the degree and order from the total number of Gauss coefficients
    Input   : totalGC is the total number of coefficients per year
    Output  : l is the Gauss coefficient degree
    """
    
    assert type(totalGC) == int
    
    countGC = totalGC
    l       = 1
    while countGC > 0:
        countGC = countGC - (2*l+1)
        l       +=1
    l       = l-1
    
    assert countGC == 0, 'The number of cofficients does not equal 2l+1 amounts'
    print('The degree and order is' , l )
    
    return l

# ### get_GClist: Create Gauss coefficient list with elements as key strings
# * The first for loop goes through each l degree
# * The second loop within the first loop creates each m values for each g and h

def get_GClist(degree, totalGC):
    """Creates a list of Gauss coefficients labels for the dictionary.
    Input   : degree
            : totalGC is the total number of coefficients
    Output  : GClist is the list of coefficients in order as strings
    """

    assert (type(degree) and type(totalGC)) == int, 'Inputs are not integers'

    GClist = ['year']
    j      = 0
    for index in range(1, degree+1):
        degreeLen = 2*index+1
        m         = 0

        for minutes in range(0, degreeLen):
            if minutes       == 0:
                placement = 'g'     
            elif minutes     == 1:
                placement = 'g'
                m += 1
            elif (minutes%2) == 0:
                placement = 'h'    
            else:
                placement = 'g'
                m += 1
            j += 1
            
            GClist.append(placement+str(index)+'_'+str(m))

    assert len(GClist) == totalGC+1
    return GClist

# ### get_dictionary: creates the dictionary
# * The GClist has the labels for the keys that are in the correct order
# * The dictionary is filled in with the columns from the 2D array of coefficients

def get_dictionary(GC_2D, GClist):
    """Creates a dictionary from the GClist keys and fills in the column from the 2D array
    Input   : GCtxt is the Gauss coefficients
            : GClist is the list of labels for the coefficients
    Output  : GCdict is the Gauss coefficients dictionary
    """

    assert type(GC_2D)   == np.ndarray
    assert type(GClist)  == list

    yearCount, numOfCols = GC_2D.shape
    totalGC              = numOfCols-1

    GCdict = {"year" : GC_2D[:,0]}
    for index in range(1, totalGC+1):
        GCdict[GClist[index]] = GC_2D[: , index]

    assert len(GCdict)                    == totalGC+1, 'The dictionary is not the size of the number of coefficients'
    assert len(np.unique(GCdict['year'])) == yearCount, 'There are duplicate years'    
    
    if len(np.unique(np.diff(GCdict['year'], n=1))) > 1:
        print('Years are not equally spaced')
    
    return GCdict 

# ### ndarray_to_string: Converts dctionary elements from ndarray to strings

def ndarray_to_string(GCdict, GClist):
    """Convert elements in the dictionary to strings so it can be read by json
    Input   : GCdict is a dictionary of the Gauss coefficients
    Input   : GClist is the list of labels for the coefficients
    Output  : GCdict_str is the dictionary with elements as strings
    """

    GCdict_str = {GClist[0] : np.array2string(GCdict[GClist[0]], max_line_width = None)}
    for index in range(1, len(GClist)):
        GCdict_str[GClist[index]] = np.array2string(GCdict[GClist[index]])

    return GCdict_str
    
# ### print_stats: prints the statistics of the dipole coefficients

def print_stats(GCdict):
    """Prints the standard deviation, mean, mininimum, and maximum of
        g1_0, g1_1, and h1_1.
    Input   : GCdict is a dictionary of the Gauss coefficients
    Output  : printed lines with statistics
    """
    
    print('g1_0 min:',np.min(GCdict['g1_0']),'mean:',np.mean(GCdict['g1_0']),'max:',np.max(GCdict['g1_0']),'std:',np.std(GCdict['g1_0']))
    print('g1_1 min:',np.min(GCdict['g1_1']),'mean:',np.mean(GCdict['g1_1']),'max:',np.max(GCdict['g1_1']),'std:',np.std(GCdict['g1_1']))
    print('h1_1 min:',np.min(GCdict['h1_1']),'mean:',np.mean(GCdict['h1_1']),'max:',np.max(GCdict['h1_1']),'std:',np.std(GCdict['h1_1']))

if __name__ == '__main__':
    main()