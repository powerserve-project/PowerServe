import time

from common import *
from tqdm import tqdm


for params in tqdm(untested_params):
    run(params)
    time.sleep(5)
