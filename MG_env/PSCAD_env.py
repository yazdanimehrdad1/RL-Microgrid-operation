import sys
import os
import shutil

# from pathlib import path

import mhi.pscad

import random
# import torch
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from statistics import mean
# %matplotlib inline

def PSCAD_env_read():
    print("PSCAD_env was read successfully")

def start_PSCAD(runTimeFilePath, resultPath, projectName):
    # if os.path.exists(resultPath) and os.path.isdir(resultPath):
    #     shutil.rmtree(resultPath)

    try:
        os.mkdir(resultPath)
    except OSError as error:
        print(error)

    pscad = mhi.pscad.launch()


def run_PSCAD(runTimeFilePath, resultPath, projectName,outputFileName, snapFileName, readFromSnap , actions, loads , BESS_1_SOC_prev_state ):
    print("actions in the PSCAD_env ", actions)
    bess1Val_P, bess1Val_Q, bess2Val_P, bess2Val_Q = actions
    load1_phaseA_Val_p,load1_phaseB_Val_p,load1_phaseC_Val_p, load1_phaseA_Val_q,load1_phaseB_Val_q,load1_phaseC_Val_q = loads

    pscad = mhi.pscad.application() #.launch()

    # pscad.load(r"C:\Users\AD-MYazdani\Desktop\SDSU Research-RL-Microgrid\MG_env\feeder4_BESS_CSI.pscx")
    print(runTimeFilePath)
    testFilePath = os.path.join(runTimeFilePath, 'feeder4_BESS_CSI.pscx')
    pscad.load(testFilePath)


    # feeder4_bess_csi = pscad.project("feeder4_BESS_CSI")
    # p_bess = feeder4_bess_csi.component(1624275686)
    # p_bess.value(3.000000)



    if not readFromSnap:
        outputFileName = 'output_steadyState.out'
        snapFileName = 'Snap_steadyState.out'
        startType = "0"

        with mhi.pscad.application() as pscad:
            mainProject = pscad.project(projectName)
            p_bess = mainProject.component(1624275686)
            p_bess.value(bess1Val_P)
            p_bess.value(bess1Val_Q)
            mainProject.parameters(time_duration="2", time_step="100", sample_step="250",
                                   StartType=startType, PlotType="1", output_filename=outputFileName,
                                   SnapType="1", SnapTime="2", snapshot_filename=snapFileName

                                   )
            mainProject.run()
    else:


        startType = "1"
        snap_fileName_read = os.path.join(runTimeFilePath+'feeder4_BESS_CSI.gf46','snap_currentState.snp')
        with mhi.pscad.application() as pscad:

            mainProject = pscad.project(projectName)


            ############################################# set BESS 1 controls#####################################
            p_bess = mainProject.component(1624275686)
            p_bess.value(bess1Val_P)


            BESS_1_SOC = mainProject.component(1703055507)
            BESS_1_SOC.parameters(Name="", Value=BESS_1_SOC_prev_state, Dim="1", )

            ############################################# set BESS 2 controls#####################################
            #p_bess = mainProject.component(1624275686)
            #fixed_load1 = mainProject.component(1013889559)
            # BESS_1_SOC = mainProject.component(1703055507)
            # BESS_1_SOC.parameters(Name="", Value=BESS_1_SOC_prev_state, Dim="1", )



            ############################################# set load 1 P and Q #####################################
            fixed_load1_phaseA = mainProject.component(229976937)
            fixed_load1_phaseB = mainProject.component(687657312)
            fixed_load1_phaseC = mainProject.component(1013889559)
            fixed_load1_phaseA.parameters(PO=load1_phaseA_Val_p, QO=load1_phaseA_Val_q)
            fixed_load1_phaseB.parameters(PO=load1_phaseB_Val_p, QO=load1_phaseB_Val_q)
            fixed_load1_phaseC.parameters(PO=load1_phaseC_Val_p, QO=load1_phaseC_Val_q)

            mainProject.parameters(time_duration="2", time_step="100", sample_step="250",
                                   StartType=startType,
                                   startup_filename=snap_fileName_read,
                                   PlotType="1", output_filename= outputFileName,
                                   SnapType="1", SnapTime="2", snapshot_filename=snapFileName

                                   )
            mainProject.run()



    # # if os.path.exists(resultPath) and os.path.isdir(resultPath):
    # #     shutil.rmtree(resultPath)
    #
    # try:
    #     os.mkdir(resultPath)
    # except OSError as error:
    #     print(error)
    #

