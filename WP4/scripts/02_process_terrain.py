"""WP4: Compute Iberian terrain statistics from buffered COP30 blocks.

Run from the repository root:
  python WP4/scripts/02_process_terrain.py --limit 1
  python WP4/scripts/02_process_terrain.py

All blocks use the same 30 m EPSG:3035 pixel lattice. Each output pixel
belongs to exactly one geographic core block; downloaded halos only supply
neighbors for interpolation/gradients. Streaming summaries are accumulated
into one checkpoint per block and combined only after all blocks succeed.
Slope p90 is computed from 0.1-degree-wide histograms (approximation <=0.1 deg).
Requires: numpy pandas rasterio geopandas shapely pyarrow scipy.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

RES = 30.0
CRS = 'EPSG:3035'
NBINS = 900  # 0.1 degree bins for slope from 0 to 90
REQUIRED = ['cell_id','elev_count','slope_count','aspect_count','elev_sum','elev_sumsq',
            'slope_sum','south_count','sin_sum','cos_sum']


def block_summary(entry, src_path, cells, res=RES):
    w, s, e, n = (float(entry[k]) for k in ['west','south','east','north'])
    # Segment edges so a geographic rectangle projects as a curved boundary.
    from shapely.geometry import Polygon
    steps = 30
    points = ([(x,s) for x in np.linspace(w,e,steps)] +
              [(e,y) for y in np.linspace(s,n,steps)] +
              [(x,n) for x in np.linspace(e,w,steps)] +
              [(w,y) for y in np.linspace(n,s,steps)])
    core_ll = Polygon(points)
    convert = Transformer.from_crs('EPSG:4326',CRS,always_xy=True).transform
    core = shapely_transform(convert, core_ll)
    x0,y0,x1,y1 = core.bounds
    # Shared projection pixel lattice: multiples of RES in both directions.
    # 120 m padding gives room for gradient computation near core edges.
    pad = 120
    left = np.floor((x0-pad)/res)*res
    top = np.ceil((y1+pad)/res)*res
    right = np.ceil((x1+pad)/res)*res
    bottom = np.floor((y0-pad)/res)*res
    width = int(round((right-left)/res))
    height = int(round((top-bottom)/res))
    transform = from_origin(left,top,res,res)
    print(f'  projected {width} x {height} ({width*height/1e6:.1f}M pixels)', flush=True)
    dem = np.empty((height,width),np.float32)
    dem.fill(np.nan)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f'Missing CRS: {src_path}')
        reproject(source=rasterio.band(src,1),destination=dem,
                  src_transform=src.transform,src_crs=src.crs,src_nodata=src.nodata,
                  dst_transform=transform,dst_crs=CRS,dst_nodata=np.nan,
                  resampling=Resampling.bilinear,num_threads=2,
                  init_dest_nodata=True)
    valid = np.isfinite(dem)
    # A geographic core mask, assigned by projected pixel centers.
    core_mask = rasterize([(core,1)],out_shape=dem.shape,transform=transform,
                          fill=0,dtype='uint8',all_touched=False).astype(bool)
    local = cells[cells.intersects(core)].reset_index(drop=True)
    if local.empty:
        raise ValueError('Manifest lists no matching cells')
    code = rasterize(((geom,i+1) for i,geom in enumerate(local.geometry)),
                     out_shape=dem.shape,transform=transform,fill=0,dtype='int16')
    assign = core_mask & (code>0) & valid
    nc = len(local)
    codes = code[assign].astype(np.int32)-1
    elev = dem[assign].astype(np.float64)
    elev_count = np.bincount(codes,minlength=nc).astype(np.int64)
    elev_sum = np.bincount(codes,weights=elev,minlength=nc)
    elev_sumsq = np.bincount(codes,weights=elev*elev,minlength=nc)
    del elev,codes,assign
    # np.gradient requires each side of the pixel. Do not include invalid
    # neighbors; the download halo makes that possible at core edges.
    from scipy.ndimage import binary_erosion
    inner = binary_erosion(valid,structure=np.ones((3,3),bool),border_value=0)
    temp = np.where(valid,dem,0).astype(np.float32,copy=False)
    dz_row,dz_x = np.gradient(temp,res,res)
    slope = np.rad2deg(np.arctan(np.hypot(dz_x,dz_row)))
    # Aspect is azimuth clockwise from North, for downslope direction.
    # row direction points south; downhill east=-dz_x, downhill north=dz_row
    aspect = (np.degrees(np.arctan2(-dz_x,dz_row))+360)%360
    good = core_mask & (code>0) & inner
    scode = code[good].astype(np.int32)-1
    sval = slope[good]
    slope_count = np.bincount(scode,minlength=nc).astype(np.int64)
    slope_sum = np.bincount(scode,weights=sval,minlength=nc)
    # Composite index for a fixed-bin global histogram. Core blocks are
    # disjoint; merging histograms yields a true GLOBAL p90 approximation.
    bins = np.minimum((sval*10).astype(np.int16),NBINS-1)
    hflat = np.bincount(scode.astype(np.int64)*NBINS + bins,
                        minlength=nc*NBINS).reshape(nc,NBINS)
    del scode,sval,bins
    directional = good & (slope>=1.0)
    acode = code[directional].astype(np.int32)-1
    angles = aspect[directional]
    arad = np.deg2rad(angles)
    aspect_count = np.bincount(acode,minlength=nc).astype(np.int64)
    south_count = np.bincount(acode,weights=((angles>=135)&(angles<225)).astype(np.float64),minlength=nc)
    sin_sum = np.bincount(acode,weights=np.sin(arad),minlength=nc)
    cos_sum = np.bincount(acode,weights=np.cos(arad),minlength=nc)
    frame = pd.DataFrame({'cell_id':local.cell_id.to_numpy(),
                          'elev_count':elev_count,'slope_count':slope_count,
                          'aspect_count':aspect_count,'elev_sum':elev_sum,
                          'elev_sumsq':elev_sumsq,'slope_sum':slope_sum,
                          'south_count':south_count,'sin_sum':sin_sum,
                          'cos_sum':cos_sum})
    return frame,hflat


def combine(manifest, checkpoints, cells, output, res):
    n=len(cells)
    index=pd.Index(cells.cell_id.to_numpy(),name='cell_id')
    totals=pd.DataFrame(0.0,index=index,columns=REQUIRED[1:])
    hist=np.zeros((n,NBINS),dtype=np.int64)
    for row in manifest:
        name=row['block']
        f=checkpoints/(name+'.parquet')
        h=checkpoints/(name+'.npz')
        if not f.exists() or not h.exists():
            raise FileNotFoundError(f'Incomplete block: {name}')
        df=pd.read_parquet(f).set_index('cell_id')
        if not df.index.is_unique:
            raise ValueError(f'Duplicate cell IDs in block {name}')
        totals.loc[df.index,REQUIRED[1:]] += df[REQUIRED[1:]].to_numpy()
        pos=index.get_indexer(df.index)
        if (pos<0).any(): raise ValueError('Unknown cell IDs')
        with np.load(h) as z:
            vals=z['hist']
        if vals.shape != (len(df),NBINS):
            raise ValueError(f'Histogram mismatch for {name}')
        hist[pos] += vals
    c=totals['elev_count'].to_numpy()
    s=totals['slope_count'].to_numpy()
    a=totals['aspect_count'].to_numpy()
    def div(x,y): return np.divide(x,y,out=np.full_like(x,np.nan,dtype=np.float64),where=y>0)
    mean=div(totals.elev_sum.to_numpy(),c)
    variance=div(totals.elev_sumsq.to_numpy(),c)-mean**2
    std=np.sqrt(np.maximum(variance,0))
    csum=np.cumsum(hist,axis=1)
    target=np.ceil(s*0.9).astype(np.int64)
    p90_bin=np.argmax(csum>=target[:,None],axis=1)
    p90=np.where(s>0,(p90_bin+0.5)/10.0,np.nan)
    cov=c*res**2/(cells.cell_area_ha.to_numpy()*1e4)
    out=pd.DataFrame({'cell_id':cells.cell_id.to_numpy(),
        'elev_mean':mean,'elev_std':std,
        'slope_mean':div(totals.slope_sum.to_numpy(),s), 'slope_p90':p90,
        'aspect_south_share':div(totals.south_count.to_numpy(),a),
        'aspect_resultant':div(np.hypot(totals.sin_sum.to_numpy(),totals.cos_sum.to_numpy()),a),
        'n_elev_pixels':c.astype('int64'),'n_slope_pixels':s.astype('int64'),
        'n_aspect_pixels':a.astype('int64'),
        'dem_coverage_frac_cell':np.clip(cov,0,1),
        'dem_coverage_note':'fraction of whole cell; coastal/ocean cells may be below 1'})
    output.parent.mkdir(parents=True,exist_ok=True)
    out.to_parquet(output,index=False)
    print(f'WROTE {len(out):,} study cells: {output}')
    print('Cells with DEM data:',int((out.n_elev_pixels>0).sum()))
    print('No DEM data:',int((out.n_elev_pixels==0).sum()))
    print('Coverage summary (whole cell denominator):')
    print(out.dem_coverage_frac_cell.describe().to_string())
    print('Missing feature counts:')
    print(out[['elev_mean','elev_std','slope_mean','slope_p90','aspect_south_share','aspect_resultant']].isna().sum().to_string())
    print('Note: coastal cells can have lower coverage because the denominator includes sea.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('WP4/outputs/dem_blocks_manifest.csv'))
    p.add_argument('--dem-dir',type=Path,default=Path('WP4/data/raw/dem/blocks'))
    p.add_argument('--cells',type=Path,default=Path('wp2/outputs/cells.gpkg'))
    p.add_argument('--checkpoints',type=Path,default=Path('WP4/outputs/terrain_checkpoints'))
    p.add_argument('--out',type=Path,default=Path('WP4/outputs/terrain_cell.parquet'))
    p.add_argument('--limit',type=int,default=0,help='Process at most N blocks, without final aggregation')
    p.add_argument('--resolution',type=float,default=RES)
    p.add_argument('--rebuild',action='store_true')
    a=p.parse_args()
    if a.resolution<=0: p.error('resolution must be positive')
    with a.manifest.open(newline='',encoding='utf-8') as f: manifest=list(csv.DictReader(f))
    if not manifest: raise ValueError('Empty manifest')
    cells=gpd.read_file(a.cells)
    cells=cells[cells.in_mask==1].to_crs(CRS).reset_index(drop=True)
    if cells.empty: raise ValueError('No in-mask cells')
    a.checkpoints.mkdir(parents=True,exist_ok=True)
    selected=manifest[:a.limit] if a.limit else manifest
    for j,row in enumerate(selected,1):
        name=row['block']
        dem=a.dem_dir/row['file']
        frame=a.checkpoints/(name+'.parquet')
        hfile=a.checkpoints/(name+'.npz')
        if frame.exists() and hfile.exists() and not a.rebuild:
            print(f'[{j}/{len(selected)}] SKIP {name}',flush=True)
            continue
        if not dem.exists(): raise FileNotFoundError(f'Missing DEM {dem}')
        print(f'[{j}/{len(selected)}] PROCESS {name}',flush=True)
        df,hist=block_summary(row,dem,cells,res=a.resolution)
        # Write temporary files then replace; incomplete blocks are reprocessed.
        ftmp=frame.with_suffix('.parquet.partial')
        htmp=hfile.with_suffix('.npz.partial')
        df.to_parquet(ftmp,index=False)
        with htmp.open('wb') as f: np.savez_compressed(f,hist=hist)
        ftmp.replace(frame); htmp.replace(hfile)
        print(f'  checkpoint: {len(df)} intersecting cells, {int(df.elev_count.sum()):,} pixels',flush=True)
    if a.limit:
        print('Pilot block complete. Re-run without --limit for all blocks and final output.')
    else:
        combine(manifest,a.checkpoints,cells,a.out,a.resolution)

if __name__=='__main__':
    main()
