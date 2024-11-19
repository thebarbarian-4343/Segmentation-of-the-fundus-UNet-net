from skimage import morphology
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import numpy as np

def skeleton_extraction_FMZ12(imm):
    #input: 
    #   imm = maschera binaria automatica (prodotta dalla rete)
    #output: 
    #   N_J2E = numero di junction to end 
    #   N_J2J = numero di junction to junction 


    #Dimensioni dell'immagine di ingresso
    height,width=np.shape(imm)
    #Definizione dei fattori di correzione basati sulle dimensioni della immagine di ingresso
    factor_w=int(width//512)
    factor_h=int(height//512)
    
    #Calcolo della IOD alla sinistra dell'immagine (sin) e alla destra (dx)
    sin=np.sum(imm[:,:width//2])
    dx=np.sum(imm[:,width//2:])

    #copio l'immagine di ingresso in un'altra matrice
    fusion=imm.copy()

    #se sin > dx allora il disco oculare è a sinistra e faccio la dilatazione a destra, altrimenti faccio l'opposto
    if sin>dx:
        #Applico la dilatazione negli ultimi 2/5 di immagine
        fusion[:,(width*3)//5:]=morphology.dilation(imm[:,(width*3)//5:],footprint=np.ones((8*factor_h,8*factor_w)))

    else:
        #Applico la dilatazione nei primi 2/5 di immagine
        fusion[:,:(width*2)//5]=morphology.dilation(imm[:,:(width*2)//5],footprint=np.ones((8*factor_h,8*factor_w)))

    #estrazione dello skeleton dall'immagine ottenuta
    skt=skeletonize(fusion)

    #Estrazione delle metriche
    skeleton_data = Skeleton(skt)
    skel_summary = summarize(skeleton_data)
    N_J2E = len(skel_summary[skel_summary['branch-type'] == 1])
    N_J2J = len(skel_summary[skel_summary['branch-type'] == 2])

    return N_J2E, N_J2J