import requests
from bs4 import BeautifulSoup
import re

from pathlib import Path

import time


def download_images():

    web_locations = ["https://sbnarchive.psi.edu/pds3/dawn/fc/DWNXFC2_1A/DATA/FITS/20071203_PERFORMANCE/",
                     "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNXFC2_1A/DATA/FITS/20100720_CHKOUT_VIRGEOMCAL/",
                     "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1A/DATA/FITS/2011123_APPROACH/2011123_OPNAV_001/",
                     "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1A/DATA/FITS/2011123_APPROACH/2011165_OPNAV_007/",
                     "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1A/DATA/FITS/2011123_APPROACH/2011198_OPNAV_017/",
                     "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1A/DATA/FITS/2011123_APPROACH/2011218_OPNAV_023/"]

    local_locations = ["cal1",
                       "cal2",
                       "opnav/2011123_OPNAV_001/",
                       "opnav/2011165_OPNAV_007/",
                       "opnav/2011198_OPNAV_017/",
                       "opnav/2011218_OPNAV_023/"]

    filt = re.compile("FC2.*F1.*")

    for addr, dest in zip(web_locations, local_locations):
        page = requests.get(addr)
        soup = BeautifulSoup(page.text)

        local_dir = Path(dest)
        local_dir.mkdir(exist_ok=True, parents=True)

        for p in soup.find_all('a', href=filt):

            if filt.match(p['href']):
                start = time.time()

                local_file = local_dir / p['href']

                r = requests.get(addr+p['href'], stream=True)
                if r.status_code == 200:
                    with local_file.open('wb') as f:
                        for chunk in r:
                            f.write(chunk)

                print('{} complete in {:.3f}'.format(p['href'], time.time()-start), flush=True)


def download_spice():
    base_url = "http://naif.jpl.nasa.gov/pub/naif/DAWN/kernels/"

    files = ['lsk/naif0012.tls',
             'pck/pck00008.tpc',
             'spk/de432.bsp',
             'spk/sb_vesta_ssd_120716.bsp',
             'pck/dawn_vesta_v02.tpc',
             'fk/dawn_v14.tf',
             'fk/dawn_vesta_v00.tf',
             'sclk/DAWN_203_SCLKSCET.00090.tsc',
             'spk/dawn_rec_070927-070930_081218_v1.bsp',
             'spk/dawn_rec_070930-071201_081218_v1.bsp',
             'spk/dawn_rec_071201-080205_081218_v1.bsp',
             'spk/dawn_rec_100208-100316_100323_v1.bsp',
             'spk/dawn_rec_100316-100413_100422_v1.bsp',
             'spk/dawn_rec_100413-100622_100830_v1.bsp',
             'spk/dawn_rec_100622-100824_100830_v1.bsp',
             'spk/dawn_rec_100824-101130_101202_v1.bsp',
             'spk/dawn_rec_101130-110201_110201_v1.bsp',
             'spk/dawn_rec_101130-110419_pred_110419-110502_110420_v1.bsp',
             'spk/dawn_rec_101130-110606_pred_110606-110628_110609_v1.bsp',
             'spk/dawn_rec_110201-110328_110328_v1.bsp',
             'spk/dawn_rec_110328-110419_110419_v1.bsp',
             'spk/dawn_rec_110328-110419_110420_v1.bsp',
             'spk/dawn_rec_110416-110802_110913_v1.bsp',
             'spk/dawn_rec_110802-110831_110922_v1.bsp',
             'spk/dawn_rec_110831-110928_111221_v1.bsp',
             'spk/dawn_rec_110928-111102_111221_v1.bsp',
             'spk/dawn_rec_110928-111102_120615_v1.bsp',
             'spk/dawn_rec_111102-111210_120618_v1.bsp',
             'spk/dawn_rec_111211-120501_120620_v1.bsp',
             'ck/dawn_fc_v3.bc',
             'ck/dawn_sc_071203_071209.bc',
             'ck/dawn_sc_071210_071216.bc',
             'ck/dawn_sc_071217_071223.bc',
             'ck/dawn_sc_071224_071230.bc',
             'ck/dawn_sc_071231_080106.bc',
             'ck/dawn_sc_100705_100711.bc',
             'ck/dawn_sc_100712_100718.bc',
             'ck/dawn_sc_100719_100725.bc',
             'ck/dawn_sc_100726_100801.bc',
             'ck/dawn_sc_110502_110508.bc',
             'ck/dawn_sc_110509_110515.bc',
             'ck/dawn_sc_110516_110522.bc',
             'ck/dawn_sc_110523_110529.bc',
             'ck/dawn_sc_110530_110605.bc',
             'ck/dawn_sc_110606_110612.bc',
             'ck/dawn_sc_110613_110619.bc',
             'ck/dawn_sc_110620_110626.bc',
             'ck/dawn_sc_110627_110703.bc',
             'ck/dawn_sc_110704_110710.bc',
             'ck/dawn_sc_110711_110717.bc',
             'ck/dawn_sc_110718_110724.bc',
             'ck/dawn_sc_110725_110731.bc',
             'ck/dawn_sc_110801_110807.bc',
             'ck/dawn_sc_110808_110814.bc',
             'ck/dawn_sc_110815_110821.bc',
             'ck/dawn_sc_110822_110828.bc',
             'ck/dawn_sc_110829_110904.bc',
             'dsk/old_versions/vesta_gaskell_512_110825.bds'
             ]

    for file in files:
        start = time.time()

        local = Path('kernels').joinpath(file)

        local.parent.mkdir(exist_ok=True, parents=True)
        
        url = base_url + file
        
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with local.open('wb') as f:
                for chunk in r:
                    f.write(chunk)

        print('{} done in {:.3f}'.format(file, time.time()-start), flush=True)


if __name__ == '__main__':
    download_spice()
    download_images()
