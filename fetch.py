import requests
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"3406e87c3f92f5c6dd80eff1c501a0a5 "}

'''
API Fetching helper functions
'''


def get(path, params=None, headers=headers):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string


    return r

def get_sim(path, params=None, headers=headers):

    r_sim = get(path, params, headers=headers)['simulations']

    return r_sim

def names(r):

    names = [sim['name'] for sim in r]
    #print(names)

    return names

def urls(r):
    urls = [sim['url'] for sim in r]
    #print(urls)

    return urls

def get_subhalo_ids(subhalo_id, snap_num=99, LBox=None, sim_name='TNG100-1'):
    remainder   = subhalo_id%100
    offset      = subhalo_id - remainder
    LBox        = 0 if LBox is None else LBox

    r           = get(baseUrl)['simulations']

    names       = [sim['name'] for sim in r]
    i           = names.index(sim_name)

    urls        = [sim['url'] for sim in r]
    sim         = get(urls[i])

    snaps       = get(sim['snapshots'])
    snap        = get(snaps[snap_num]['url'])

    subhalos    = get(snap['subhalos'], {'order_by':'id', "offset": offset})
    subhalo     = get(subhalos['results'][remainder]['url'])
    mpb         = get(subhalo['trees']['sublink_mpb'])

    with h5py.File(mpb, 'r') as f:
        pos = f['SubhaloPos'][:]
        snapnum = f['SnapNum'][:]
        subID = f['SubfindID'][:]

    pos = unfold_periodic_positions(np.flip(pos, axis=0), LBox)
    snapnum = np.flip(snapnum, axis=0)

    os.remove(mpb)

    # Pos to check
    return pos, snapnum, subID

def unfold_periodic_positions(periodic_positions_history, LBox=None):
    if LBox is None:
        return periodic_positions_history.copy()

    diffs = np.diff(periodic_positions_history, axis=0)

    # Correct for periodic boundary crossing
    diffs[diffs > LBox / 2] -= LBox
    diffs[diffs < -LBox / 2] += LBox

    # Reconstruct absolute positions by cumulative sum
    absolute_positions = np.vstack([
        periodic_positions_history[0],
        periodic_positions_history[0] + np.cumsum(diffs, axis=0)
    ])


    return absolute_positions


if __name__ == "__main__":
    # Code used to test fetching features
    r = get(baseUrl)['simulations']

    names = [sim['name'] for sim in r]
    i = names.index('TNG50-1')

    urls = [sim['url'] for sim in r]
    sim1 = get(urls[i])
    snaps = get(sim1['snapshots'])

    print(snaps[0].keys())

    last_snap = get(snaps[99]['url'])

    print(last_snap['url'])


    subhalos = get(last_snap['subhalos'], {'order_by':'id', "offset": 546100})

    '''
    subhalo = get(subhalos['results'][-1]['url'])
    #print(subhalo.keys())

    #print(subhalo['trees'].keys())

    mpb1 = get(subhalo['trees']['sublink_mpb'])
    print(mpb1)


    f = h5py.File(mpb1,'r')
    #print(f.keys())
    #print(len(f['SnapNum']))
    #print(f['SnapNum'][:])
    f.close()

    os.remove(mpb1)


    plt.plot(snapnum, subid, label='subID')
    plt.legend()
    plt.xlabel('Snapshot Number')
    plt.ylabel('Subhalo ID')

    plt.show()
    '''
