import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import illustris_python as il
from classes import Halo, Subhalo
from metrics import *
from fetch import *
from halo_data_folder_organiser import organize_halo_folders

basePath = r"D:/Abhilash_Thesis_Data/Simulation_Data/TNG100-1/output"
sim_name = 'TNG100-1'

'''
Default script for reading/extracting data from TNG-100-1
'''


# Exported 68-97, 99
# Downloaded 68-99
snap_nos = [99]

for snap_no in snap_nos:
    snap_name = "Snap"+str (snap_no)+"_100"

    subhalo_fields = ['SubhaloFlag', 'SubhaloBHMdot', 'SubhaloPos', 'SubhaloParent', 'SubhaloGrNr', 'SubhaloMass', 'SubhaloStellarPhotometrics', 'SubhaloStarMetalFractions', 'SubhaloVel']
    subhalos = il.groupcat.loadSubhalos(basePath, snap_no, subhalo_fields)

    halo_fields = ['GroupCM', 'GroupPos', 'GroupFirstSub', 'GroupNsubs', 'GroupMass', 'Group_R_Crit200']
    halos = il.groupcat.loadHalos(basePath, snap_no, halo_fields)

    print(len(halos['GroupFirstSub']))

    header = il.groupcat.loadHeader(basePath, snapNum=snap_no)
    LBox = header['BoxSize']

    # Subhalo deciding criteria
    #group_catalog_categorisation = True
    group_catalog_categorisation = False

    halo_relevance_condition_lower = np.where(halos['GroupMass'] > 50)[0]
    halo_relevance_condition_upper = np.where(halos['GroupMass'] < 500)[0]
    halo_relevance_condition = np.intersect1d(halo_relevance_condition_lower, halo_relevance_condition_upper)
    relevant_halo_indices = []

    central_subhalo_indices = halos['GroupFirstSub'][halo_relevance_condition]
    relevant_halos = []

    print(len(halo_relevance_condition))

    subhalo_pos = subhalos['SubhaloPos']

    #Relevant systems
    if group_catalog_categorisation:
        for halo in tqdm(halo_relevance_condition):
            possibly_relevant_halo = Halo(halos, halo, halo_fields, LBox, snap_no, sim_name)
            possibly_relevant_halo.get_subhalos(subhalos, subhalo_fields)
            if possibly_relevant_halo.metrics['visible_subhalos'] > 4:
                relevant_halos.append(possibly_relevant_halo)
            else:
                del possibly_relevant_halo

    else:
        # Decide subhalo by volume
        for i, halo in tqdm(enumerate(halo_relevance_condition), total=len(halo_relevance_condition)):
            possibly_relevant_halo = Halo(halos, halo, halo_fields, LBox, snap_no, sim_name)
            central_subhalo_index = central_subhalo_indices[i]
            possibly_relevant_halo.get_subhalos_by_volume(subhalos, subhalo_fields, central_subhalo_index, subhalo_pos)
            if possibly_relevant_halo.metrics['visible_subhalos'] > 4:
                relevant_halos.append(possibly_relevant_halo)
                relevant_halo_indices.append(halo)
            else:
                del possibly_relevant_halo

    print(len(relevant_halos))

    j = 0
    sats = 0
    relevant_halo_mass_indices = []

    for i, relevant_halo in enumerate(relevant_halos):
        relevant_halo.calc_metrics()
        central_galaxy_mass = relevant_halo.metrics['Central_Galaxy_Mass']
        if (central_galaxy_mass>=50 and central_galaxy_mass<=300): # and (relevant_halo.metrics['centroid']) < 0.05:
            j += 1
            sats += relevant_halo.metrics['visible_subhalos']
            '''
            relevant_halo_mass_indices.append(int(relevant_halo_indices[i]))
            metrics = "{}. {}".format(relevant_halo_indices[i], relevant_halo.metrics)
            print(metrics)

            #relevant_halo.track_visibile_subhalos()
            #relevant_halo.correct_tracks()
            
            #relevant_halo.calc_metrics_over_time_using_tracks()
            #relevant_halo.process_top_10()
            #relevant_halo.calc_top_10_metrics_over_time()
            #relevant_halo.analyze_merger_history(basePath=basePath)


            relevant_halo.save(outdir=snap_name, 
                               save_metrics=True, save_subhalo_pos=True, save_dark_subhalo=True, save_tracks=False,
                               save_metric_evolution_using_tracks=False, save_central_ids=False,
                               save_top_10=False, save_merger_analysis=False)#, save_metric_evolution_using_tracks=False)
            #relevant_halo.save(outdir="Save_TNG_100-1_by_volume_PBC",
            #                   save_metrics=False, save_subhalo_pos=False, save_dark_subhalo=False, save_tracks=False, save_metric_evolution_using_tracks=False, save_central_ids=True)

            # Check to rerun only specific Halo systems
            #j += 1
            # print(j)
            #if j == 5:
                #break
            '''

    #print(relevant_halo_mass_indices)
    #print(len(relevant_halo_mass_indices))
    # print()
    print(j)
    print(sats)

    '''
    # ---------------------------------------------------------
    # AUTOMATIC FOLDER ORGANIZATION
    # ---------------------------------------------------------
    #target_folder = os.path.join("halo_exports", f"Save_{sim_name}_by_volume_PBC", f"Snap{snap_no}")
    target_folder = os.path.join("halo_exports", snap_name)

    organize_halo_folders(target_folder)
    '''