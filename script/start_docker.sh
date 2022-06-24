#!/bin/bash

nvidia-docker run -d --shm-size=256m --name localring -v $(pwd):/code -it teslazhu/torchgm
