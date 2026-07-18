import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pyvista as pv

def tet_vol(vert_mat):
    fourones = np.array([np.full(4,[1.])]).T
    vert_mat = np.append(vert_mat,fourones,axis=1)
    return(np.abs(np.linalg.det(vert_mat)/6))
def tet_netmag(verts,elems,mags,i):
    elem = elems[i]
    vert_mat = verts[elem]
    mag = mags[elem]
    vol = tet_vol(vert_mat)
    net_mag = vol * np.mean(mag,axis=0)
    return(net_mag)
def get_net_mag(verts,elems,mags):
    netmag = np.zeros(3)
    for i in range(len(elems)):
        netmag += tet_netmag(verts,elems,mags,i)
    return(netmag)
def T_matrix(vert_mat):
    T = (vert_mat[:-1] - vert_mat[-1]).T
    return(T)
def xyz_to_barycentric(xyz,vert_mat):
    barycentric = np.empty(4)
    T = T_matrix(vert_mat)
    xyz_diff = xyz - vert_mat[-1]
    barycentric[:-1] = np.linalg.inv(T)@xyz_diff
    barycentric[-1] = 1 - np.sum(barycentric[:-1])
    return(barycentric)


def make_LEM_summary_file(folders,TMxs,PROs,OBLs,sizes,alignments,outfile):
    
    files = []
    PRO_list = []
    OBL_list = []
    TMx_list = []
    size_list = []

    for i,folder in enumerate(folders):
        for file in os.listdir(folder):
            file = file[:6]
            if folder+file not in files:
                files.append(folder+file)
                PRO_list.append(PROs[i])
                OBL_list.append(OBLs[i])
                TMx_list.append("TM "+(str(TMxs[i]).zfill(2)))
                size_list.append(sizes[i])

    
    Mag_x = []
    Mag_y = []
    Mag_z = []
    vort_x = []
    vort_y = []
    vort_z = []
    hel = []
    e_tot = []
    e_demag = []
    e_exch = []
    e_aniso = []

    j = 0
   
    #files = np.sort(np.unique(files))
    for i,file in enumerate(files):
        try:
            print(f'Working on: {file}')
            hdf5 = h5py.File(file+".hdf5")
            verts = hdf5['mesh']["vertices"][:]
            elems = hdf5['mesh']['elements'][:]
            mags = hdf5['magnetizations']['0'][:]
            net_mag = get_net_mag(verts,elems,mags)

            if j == 0:
                cells = np.append(np.full((len(elems),1),4),elems,axis=1).astype(int).flatten()
                celltypes = np.full(len(elems),pv.CellType.TETRA)
                points = verts
                grid = pv.UnstructuredGrid(cells, celltypes, points)
                j+=1
            grid['vectors'] = mags
            grad = grid.compute_derivative(scalars='vectors',vorticity=True)
            net_vort = get_net_mag(verts,elems,grad['vorticity'])
            hels = np.sum(mags*grad['vorticity'],axis=1)
            net_hel = get_net_mag(verts,elems,hels)[0]
            normed_vort = grad['vorticity']/np.linalg.norm(grad['vorticity'],axis=1,keepdims=True)
            r_hel = np.sum(normed_vort*mags,axis=1)
            r_hel = get_net_mag(verts,elems,r_hel)[0]
            dat = pd.read_csv(file+'.log',sep='\s+',skiprows=12)
            dat = dat.drop(['E-Stress','E-Exch2','E-Exch3','E-Exch4','E-ext'],axis=1)
            energies = dat.loc[:,'E-Anis':'E-Tot'].iloc[-1].values
            with open(file+'.log') as f:
                KdV = float(f.readlines()[7].split(' ')[-1][:-1])
            energies *= KdV
            Mag_x.append(net_mag[0])
            Mag_y.append(net_mag[1])
            Mag_z.append(net_mag[2])
            vort_x.append(net_vort[0])
            vort_y.append(net_vort[1])
            vort_z.append(net_vort[2])
            hel.append(r_hel)
            e_tot.append(energies[3])
            e_demag.append(energies[2])
            e_exch.append(energies[1])
            e_aniso.append(energies[0])
            if i!= (len(files) - 1):
                if files[i+1] != files[i]:
                    j = 0
        except:
            print("Issue with this model!")
            Mag_x.append(np.nan)
            Mag_y.append(np.nan)
            Mag_z.append(np.nan)
            vort_x.append(np.nan)
            vort_y.append(np.nan)
            vort_z.append(np.nan)
            hel.append(np.nan)
            e_tot.append(np.nan)
            e_demag.append(np.nan)
            e_exch.append(np.nan)
            e_aniso.append(np.nan)

    # Alpha, theta, phi, curently hardcoded
    df = pd.DataFrame({'Unique ID':files,'Size':np.array(size_list)/1000,'Material':TMx_list,'Prolateness':PRO_list,'Oblateness':OBL_list,'Alpha':0,'Theta':0,'Phi':0,'Temperature':20,'Mag x':Mag_x,'Mag y':Mag_y,'Mag z':Mag_z,'Vorticity x':vort_x,'Vorticity y':vort_y,'Vorticity z':vort_z,'Relative helicity':hel,'Anisotropy energy':e_aniso,'Exchange energy':e_exch,'Demag energy':e_demag,'Total energy':e_tot,'Directory':files})
    df.to_csv(outfile)
    return df



folders = []
TMxs = []
PROs = []
OBLs = []
sizes = []
alignments = []


for TMx in [00]:
    for PRO in [1.44]:
        for size in [120]:
            for alignment in ['hard']:
               folder = "./LEMs/"
               if os.path.isdir(folder):
                   TMxs.append(TMx)
                   PROs.append(PRO)
                   OBLs.append(1)
                   sizes.append(size)
                   alignments.append(alignment)
                   folders.append(folder)
               else:
                    print(f"No Such Directory: {folder}")

make_LEM_summary_file(folders,TMxs,PROs,OBLs,sizes,alignments,"LEMs.csv")
