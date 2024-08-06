import numpy as np 
from skimage.draw import disk

def design_cavity(nonpmlNx, nonpmlNy, Npmlsep, circle, outer_radius_x, outer_radius_y, inner_radius):
    """
    This function returns a mask that can be used to design a cavity.
    The cavity can be a circle with a circular design region or a square with a square design region.
    """

    Mx = outer_radius_x
    My = outer_radius_y
    Mi = inner_radius
    design_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=bool)
    if circle:
        rr1, cc1 = disk((nonpmlNx//2, nonpmlNy//2), Mx//2, shape=design_mask.shape)
        design_mask[rr1, cc1] = 1
        if Mi > 0:
            rr2, cc2 = disk((nonpmlNx//2, nonpmlNy//2), Mi, shape=design_mask.shape) 
            design_mask[rr2, cc2] = 0
    else:
        design_mask[Npmlsep:Npmlsep+Mx, Npmlsep:Npmlsep+My] = True
        if Mi > 0:
            design_mask[nonpmlNx//2 - Mi : nonpmlNx//2 + 1 + Mi,  nonpmlNy//2 - Mi : nonpmlNy//2 + 1 + Mi] = False

    return design_mask