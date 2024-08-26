########################################################
# Perform dynamic analysis using Openseespy. SI unit
# 
# Dependancy: 
# - openseespy, pandas, numpy, matplotlib
########################################################

from ctypes import Union
import matplotlib.pyplot as plt
from cmath import pi
from opensees.openseespy import *
import pandas as pd
import numpy as np
import ReadRecord
from pathlib import Path
import os
import mpl_toolkits.axisartist as axisartist

class MDOFOpenSees():

    UniqueRecorderPrefix = 'URP0_'
    __g = 9.8

    # structrual parameters
    NStories : int = 0
    m: list = []
    k: list = []
    DampingRatio:float = 0.05
    HystereticCurveType: str = 'Elastic'
    HystereticParameters = ()
    SelfCenteringEnhancingFactor = 0.0 # 0-1

    # output directory
    outputdir = str(Path.cwd())

    # pushover analysis results
    # DriftHistory = {} # DriftHistory['time'] is the time list. DriftHistory[1] is the IDR list of 1st story
    # ForceHistory = {} 
    NodeDispHistory = {} # NodeDispHistory['time'], NodeDispHistory[1-N]

    # Dynamic analysis results
    MaxDrift = np.array([]) # MaxDrift[0] is the 1st story
    MaxAbsAccel = np.array([]) # MaxAbsAccel[0] is the ground
    MaxRelativeAccel = np.array([]) # [0] is the ground
    ResDrift = None
    DriftHistory = {} # DriftHistory['time'] is the time list. DriftHistory[1] is the IDR list of 1st story
    ForceHistory = {} 
    NodeAbsAccelHistory = {} # NodeAbsAccelHistory[0] is the ground
    NodeRelativeAccelHistory = {}


    def __init__(self, NStories :int, m: list, k:list, DampingRatio:float,
        HystereticCurveType: str, *HystereticParameters):
        # -NStories, number of stories
        # -m, mass list for each floor, kg
        # -k, elastic stiffness for each story, N/m
        # -DampingRatio, scalar
        # -HystereticCurveType, ['Elastic','Modified-Clough','Kinematic hardening','Pinching']
        # -*HystereticParameters, variable parameters including (Vyi, betai, etai, DeltaCi, tao)
        # {
        #   -Vyi, yield shear force for each story, N
        #   -betai, overstrength ratio of ultimate strength to yield strength for each story
        #   -etai, hardening ratio for each story
        #   -DeltaCi, displacement threshold for complete damage state, m
        #   -tao, degradation factor
        # }

        self.NStories = NStories
        self.m = m
        self.k = k
        self.DampingRatio = DampingRatio
        self.HystereticCurveType = HystereticCurveType
        self.HystereticParameters = HystereticParameters

    def StaticPushover(self, maxU: list = [0.10,-0.10,0], dU = 0.001,
        CFloor = 'roof', ifprint: bool = True):
        # Parameters:
        # maxU - target disp (m).
        # dU - Displacement increment (m)
        # CFloor - controling floor. 
        # 
        # Returns:
        # Iffinish, currentDisp

        model = self.model
        
        if ifprint:
            print('Pushover analysis of a MDOF lumped-mass building model with OpenSees...')
        
        self.__BuildModel(ifprint)

        tsTag = 301
        timeSeries('Linear', tsTag)
        patternTag = 101
        pattern('Plain', patternTag, tsTag)

        # Create nodal loads
        #    nd    FX  FY  MZ
        for i in range(1,self.NStories+1):
            model.load(i, i, 0.0, 0.0)

        # recorders
        outputdir = Path(self.outputdir).relative_to(Path.cwd())
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')), '-time',
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), '-time',
            '-ele', *list(range(1,self.NStories+1)), 'axialForce')
        recorder('Node', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'NodeDispHistory.txt')),'-time',
            '-node', *list(range(1,self.NStories+1)), '-dof', 1, 'disp')
        
        # Perform analysis        
        Tol = 1e-6
        maxNumIter = 100
        if isinstance(CFloor,str) & (CFloor == 'roof'):
            CFloor = self.NStories
        system('FullGeneral')
        constraints('Transformation')
        numberer('RCM')
        test('NormDispIncr', Tol, maxNumIter)
        algorithm('NewtonLineSearch') 

        Test = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 3:'EnergyIncr', 
            4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
        Algorithm = {1:'KrylovNewton', 2: 'SecantNewton' , 3:'ModifiedNewton' , 
            4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden', 8: 'NewtonLineSearch'}

        currentDisp = 0.0
        ok = 0

        for i in range(len(maxU)):
            while ok == 0 and abs(currentDisp-maxU[i])>dU:
                numIter=100
                integrator('DisplacementControl', CFloor, 1, 
                    np.sign(maxU[i]-currentDisp)*dU, numIter)
                analysis('Static')
                ok = analyze(1)
                # if the analysis fails try initial tangent iteration
                if ok != 0:
                    break
                currentDisp = nodeDisp(CFloor, 1)

        Iffinish = not ok

        if ifprint:
            print(f'State (Successful or Fault): {Iffinish:d}')
        
        wipe()
        self.__ReadPushoverRecorderFiles()

        return Iffinish, currentDisp
        
    def DynamicAnalysis(self, EQRecordfile:str, GMScaling:float, ifprint: bool = True,
        DeltaT = 0.1):
        # Parameters:
        # -ifprint, true or false
        # -EQRecordfile, earthquake record file which is in PEER format, such as 'H-E12140'
        # -GMScaling, ground motion scaling factor
        # -DeltaT, 'AsInRecord' or a float
        # 
        # Return:
        # Iffinish, tCurrent, TotalTime

        if ifprint:
            print('Perform dynamic analysis of a MDOF lumped-mass building model with OpenSees...')

        self.__BuildModel(ifprint)

        # Permform the conversion from SMD record to OpenSees record
        p = Path(EQRecordfile)
        dt, nPts = ReadRecord.ReadRecord(EQRecordfile, 
            (Path(p.parent, self.UniqueRecorderPrefix + p.name +'.dat')).as_posix())

        # Uniform EXCITATION: acceleration input
        tsTag = 100
        EQfile = Path(p.parent,self.UniqueRecorderPrefix + p.name +'.dat')
        timeSeries('Path', tsTag, '-dt', dt, '-filePath', 
            os.path.relpath(EQfile,Path.cwd()),
            '-factor', self.__g * GMScaling) # 用相对路径，避免路径中有中文字符
        IDloadTag = 400			# load tag
        GMdirection = 1
        pattern('UniformExcitation', IDloadTag, GMdirection, '-accel', tsTag)

        # recorders
        outputdir = Path(self.outputdir).relative_to(Path.cwd())
        recorder('EnvelopeElement', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'MaxDrift.txt')),
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')),'-time',
            '-ele', *list(range(1,self.NStories+1)), 'deformations')
        recorder('Element', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), '-time',
            '-ele', *list(range(1,self.NStories+1)), 'axialForce')
        recorder('EnvelopeNode', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'MaxAbsAccel.txt')), 
            '-timeSeries', tsTag, 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('EnvelopeNode', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'MaxRelativeAccel.txt')),
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('Node', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'NodeAbsAccelHistory.txt')),
            '-timeSeries', tsTag, '-time', 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')
        recorder('Node', '-file', 
            str(Path(outputdir,self.UniqueRecorderPrefix+'NodeRelativeAccelHistory.txt')), '-time', 
            '-node', *list(range(self.NStories+1)), '-dof', 1, 'accel')


        # dynamic analysis
        Tol = 1e-8
        maxNumIter = 10
        DtAnalysis = dt if DeltaT== 'AsInRecord' else DeltaT # dt
        wipeAnalysis()
        constraints('Transformation')
        numberer('RCM')
        # system('UmfPack') # only this works when using ExpressNewton algorithm.
        system('BandGeneral')

        tCurrent = getTime()

        Test = {1:'NormDispIncr', 2: 'RelativeEnergyIncr', 3:'EnergyIncr', 
            4: 'RelativeNormUnbalance',5: 'RelativeNormDispIncr', 6: 'NormUnbalance'}
        Algorithm = {8: 'NewtonLineSearch', 1:'KrylovNewton', 2: 'SecantNewton' , 3:'ModifiedNewton' , 
            4: 'RaphsonNewton',5: 'PeriodicNewton', 6: 'BFGS', 7: 'Broyden'} # 9: 'ExpressNewton', 

        # algorithm ExpressNewton 2 1.0 -currentTangent -factorOnce

        tFinal = nPts*dt

        time = [tCurrent]
        ok = 0
        while tCurrent < tFinal:   
            for i in Test:
                test(Test[i], Tol, maxNumIter)    
                for j in Algorithm: 
                    if j==9:
                        algorithm(Algorithm[j], 2, 1.0, '-currentTangent','-factorOnce')
                    elif j < 4:
                        algorithm(Algorithm[j], '-initial')
                    else:
                        algorithm(Algorithm[j])
                    while ok == 0 and tCurrent < tFinal:    
                        NewmarkGamma = 0.5
                        NewmarkBeta = 0.25
                        integrator('Newmark', NewmarkGamma, NewmarkBeta)
                        analysis('Transient')
                        ok = analyze(1, DtAnalysis)
                        if ok == 0:
                            tCurrent = getTime()                
                            time.append(tCurrent)
            break

        Iffinish = not ok
        TotalTime = tFinal

        if ifprint:
            print(f'State (Successful or Fault): {Iffinish:d}')
            print(f'The analysis ends at {tCurrent:.3f} sec out of {TotalTime:.3f} sec.')
        
        wipe()
        self.__ReadDynamicRecorderFiles()

        return Iffinish, tCurrent, TotalTime

    def PlotForceDriftHistory(self, NumOfStory:int = 1):
        cm = 1/2.54  # centimeters in inches
        fig = plt.figure('Origional',(10*cm,8*cm))
    
        ax = axisartist.Subplot(fig, 1,1,1)
        fig.add_axes(ax)
        
        ax.axis[:].set_visible(False)
        
        ax.axis["x"] = ax.new_floating_axis(0, 0)
        ax.axis["y"] = ax.new_floating_axis(1, 0)
        ax.axis["x"].set_axis_direction('top')
        ax.axis["y"].set_axis_direction('left')
        ax.axis["x"].set_axisline_style("->", size = 2.0)
        ax.axis["y"].set_axisline_style("->", size = 2.0)
        
        ax.plot(self.DriftHistory[NumOfStory],self.ForceHistory[NumOfStory],linewidth = 2)
        # plt.title('y = 2sin(2t)',fontsize = 14, pad = 20)
        
        # ax.set_xticks(np.linspace(0.25,1.25,5)*np.pi)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # ax.set_xticklabels(['$\\frac{\pi}{4}$','$\\frac{\pi}{2}$', '$\\frac{3\pi}{4}$', '$\pi$', '$\\frac{5\pi}{4}$', '$\\frac{3\pi}{2}$'])
        # ax.set_yticks([0, 1, 2])
        
        # ax.set_xlim(-0.5*np.pi,1.5*np.pi)
        # ax.set_ylim(-2.2, 2.2)
        
        plt.show()


    def __BuildModel(self, ifprint: bool):
        # define building model

        wipe()			
        model = self.model = ops.Model(ndm=2, ndf=3)
        

        storyLength = 1.0

        # node
        model.node(0, 0., 0.)
        model.fix(0, 1, 1, 1) 
        for i in range(self.NStories):
            model.node(i+1, (i+1)*storyLength, 0.)
            model.mass(i+1, self.m[i], 0., 0.)
            model.fix(i+1, 0, 1, 1) 
        
        # material
        E = 1.0
        matTag = [i+1 for i in range(self.NStories)]
        A = [0] * self.NStories
        for i in range(self.NStories):
            A[i] = self.k[i] * storyLength / E
            # *HystereticParameters = (Vyi, betai, etai, DeltaCi, tao)
            if self.HystereticCurveType == 'Elastic':
                uniaxialMaterial(self.HystereticCurveType, matTag[i], E)
            elif self.HystereticCurveType in ['Modified-Clough','Kinematic hardening','Pinching']:
                Vyi = self.HystereticParameters[0][i]
                betai = self.HystereticParameters[1][i]
                etai = self.HystereticParameters[2][i]
                DeltaCi = self.HystereticParameters[3][i]
                s1p = Vyi / A[i] / E  # yield stress
                e1p = s1p / E    # yield strain
                s2p = s1p * betai
                e2p = e1p + (s2p-s1p) / (etai * E)
                s3p = s2p*1.001
                e3p = DeltaCi/storyLength
                if e3p < e2p:
                    print('WARNING: the drift of complete damage is smaller than ultimate drift')
                    e2p = e3p
                    s2p = (e2p - e1p)*(etai * E) + s1p
                    s3p = s2p*1.001
                    e3p = e2p*1.1
                if self.HystereticCurveType == 'Modified-Clough':
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, 0.5, 0.5, 
                        0, 0, 0.0)
                elif self.HystereticCurveType == 'Kinematic hardening':
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, 0.001, 0.999, 
                        0.0, 0.0, 0.0)
                elif self.HystereticCurveType == 'Pinching':
                    tao = self.HystereticParameters[4]
                    if tao == 0:
                        tao = 0.001
                    elif tao == 1:
                        tao = 0.999
                    else:
                        pass
                    py = tao
                    px = 1.0 - py
                    uniaxialMaterial('Hysteretic', matTag[i], 
                        s1p, e1p, s2p, e2p, s3p, e3p, 
                        -s1p, -e1p, -s2p, -e2p, -s3p, -e3p, px, py, 
                        0, 0, 0.0)
                
                if (self.SelfCenteringEnhancingFactor > 0) & (self.SelfCenteringEnhancingFactor <= 1):
                    matTag_MultiLinear = 1000+matTag[i]
                    uniaxialMaterial('ElasticMultiLinear', matTag_MultiLinear, 
                        0.0, '-strain', -e3p,-e2p,-e1p,e1p,e2p,e3p, 
                        '-stress', -s3p,-s2p,-s1p,s1p,s2p,s3p)
                    matTag_Parallel = 2000+matTag[i]
                    uniaxialMaterial('Parallel', matTag_Parallel, matTag[i], matTag_MultiLinear, 
                        '-factors', 1.0-self.SelfCenteringEnhancingFactor,self.SelfCenteringEnhancingFactor)
            else:
                print('Error: incorrect Hysteretic Curve Type')
                return

        # element
        for i in range(self.NStories):
            if (self.SelfCenteringEnhancingFactor > 0) & (self.SelfCenteringEnhancingFactor <= 1):
                element('Truss', i+1, i,i+1, A[i], 2000+matTag[i])
            else:
                element('Truss', i+1, i,i+1, A[i], matTag[i])

        # Eigenvalue Analysis   
        if self.NStories>1:  
            lambdaN = eigen('-fullGenLapack', 2)
            w1 = lambdaN[0]**0.5
            w2 = lambdaN[1]**0.5
            T1 =  2.0*pi/w1
            T2 =  2.0*pi/w2
            if ifprint:
                print(f'Eigen Analysis: T1 = {T1:.2f} s; T2 = {T2:.2f} s')
        else:
            lambdaN = eigen('-fullGenLapack', 1)
            w1 = lambdaN[0]**0.5
            T1 =  2.0*pi/w1
            if ifprint:
                print(f'Eigen Analysis: T1 = {T1:.2f} s')

        # define & apply damping
        # RAYLEIGH damping parameters, Where to put M/K-prop damping, switches 
        # (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/1099.htm)
        # D=$alphaM*M + $betaKcurr*Kcurrent + $betaKcomm*KlastCommit + $beatKinit*$Kinitial
        if self.NStories>1: 
            xDamp = self.DampingRatio;  
            MpropSwitch = 1.0
            KcurrSwitch = 0.0
            KcommSwitch = 0.0
            KinitSwitch = 1.0
            nEigenI = 1 
            nEigenJ = 2 
            lambdaI = lambdaN[nEigenI-1] 
            lambdaJ = lambdaN[nEigenJ-1] 
            omegaI = lambdaI**0.5
            omegaJ = lambdaJ**0.5
            alphaM = MpropSwitch*xDamp*(2.0*omegaI*omegaJ)/(omegaI+omegaJ)
            betaKcurr = KcurrSwitch*2.*xDamp/(omegaI+omegaJ)      # current-K;      +beatKcurr*KCurrent
            betaKcomm = KcommSwitch*2.*xDamp/(omegaI+omegaJ)      # last-committed K;   +betaKcomm*KlastCommitt
            betaKinit = KinitSwitch*2.*xDamp/(omegaI+omegaJ)      # initial-K;     +beatKinit*Kini
            rayleigh(alphaM,betaKcurr, betaKinit, betaKcomm)       
        else:
            xDamp = self.DampingRatio;  
            MpropSwitch = 1.0
            nEigenI = 1 
            lambdaI = lambdaN[nEigenI-1] 
            omegaI = lambdaI**0.5
            alphaM = MpropSwitch*xDamp*2.0*omegaI
            rayleigh(alphaM, 0, 0, 0)  

    def __ReadDynamicRecorderFiles(self):

        # check if analysis results are empty
        fpath = str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxDrift.txt'))
        if not (os.path.isfile(fpath) and os.path.getsize(fpath) > 0):
            return

        self.MaxDrift = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxDrift.txt')), 
            sep='\s+', header=None).loc[2,:].values
        self.MaxAbsAccel = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxAbsAccel.txt')), 
            sep='\s+', header=None).loc[2,:].values
        self.MaxRelativeAccel = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'MaxRelativeAccel.txt')), 
            sep='\s+', header=None).loc[2,:].values
        
        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')), 
            sep='\s+', header=None)
        self.DriftHistory = {}
        self.DriftHistory['time'] = df.loc[:,0]
        ind_last5sec = ((self.DriftHistory['time'][-1:]-self.DriftHistory['time'])<5.0)
        ResDrift_dict = {}
        for i in range(self.NStories):
            self.DriftHistory[i+1] = df.loc[:,i+1]
            ResDrift_dict[i+1] =  self.DriftHistory[i+1][ind_last5sec].mean()

        self.ResDrift = np.abs(np.array(list(ResDrift_dict.values()))).max()

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), 
            sep='\s+', header=None)
        self.ForceHistory = {}
        self.ForceHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.ForceHistory[i+1] = df.loc[:,i+1]
        
        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'NodeAbsAccelHistory.txt')), 
            sep='\s+', header=None)
        self.NodeAbsAccelHistory = {}
        self.NodeAbsAccelHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeAbsAccelHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'NodeRelativeAccelHistory.txt')), 
            sep='\s+', header=None)
        self.NodeRelativeAccelHistory = {}
        self.NodeRelativeAccelHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeRelativeAccelHistory[i+1] = df.loc[:,i+1]

    def __ReadPushoverRecorderFiles(self):

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'DriftHistory.txt')), 
            sep='\s+', header=None)
        self.DriftHistory = {}
        self.DriftHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.DriftHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'ForceHistory.txt')), 
            sep='\s+', header=None)
        self.ForceHistory = {}
        self.ForceHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.ForceHistory[i+1] = df.loc[:,i+1]

        df = pd.read_table(
            str(Path(self.outputdir,self.UniqueRecorderPrefix+'NodeDispHistory.txt')), 
            sep='\s+', header=None)
        self.NodeDispHistory = {}
        self.NodeDispHistory['time'] = df.loc[:,0]
        for i in range(self.NStories):
            self.NodeDispHistory[i+1] = df.loc[:,i+1]
