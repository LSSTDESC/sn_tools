from sn_tools.sn_io import checkDir
import os

class BatchIt:
    """
    class to setup environment, create batch script, and launch the batch
    """
    def __init__(self, logDir='logs',scriptDir='scripts',processName='test_batch',
                 account='lsst',L='sps',time='20:00:00',mem='10G',n=8):

        self.dict_batch = {}

        self.dict_batch['--account'] = account
        self.dict_batch['-L'] = L
        self.dict_batch['--time'] = time
        self.dict_batch['--mem'] = mem
        self.dict_batch['-n'] = n

        # create output dirs if necessary
        self.checkDirs(logDir,scriptDir)

        # output files
        self.prepareOut(processName)

        # start filling script
        self.startScript()
        
    def checkDirs(self,logDir,scriptDir):
        """
        Method to create (if necessary) output dirs
        Parameters
        ---------------
        logDir: str
           log directory name
        scriptDir: str
           script directory name
        """

        # get current directory
        self.cwd = os.getcwd()

        # script dir
        self.scriptDir = '{}/{}'.format(self.cwd,scriptDir)
        checkDir(self.scriptDir)

        self.logDir = '{}/{}'.format(self.cwd,logDir)
        checkDir(self.logDir)

        

    def prepareOut(self,processName):
        """
        method to define a set of files required for batch
        Parameters
        ----------------
        processName: str
          name of the process
        """
        
        self.scriptName = '{}/{}.sh'.format(self.scriptDir,processName)
        self.logName = '{}/{}.log'.format(self.logDir,processName)
        self.errlogName =  '{}/{}.err'.format(self.logDir,processName)


    def startScript(self):

        """
        Method to write generic parameter to the script
        """

        self.dict_batch['--output'] = self.logName
        self.dict_batch['--error'] = self.errlogName

        # fill the script
        script = open(self.scriptName, "w")
        #script.write(qsub + "\n")
        script.write("#!/bin/env bash\n") 
        for key, vals in self.dict_batch.items():
            script.write("#SBATCH {} {} \n".format(key,vals))

        script.write(" cd " + self.cwd + "\n")
        script.write(" export MKL_NUM_THREADS=1 \n")
        script.write(" export NUMEXPR_NUM_THREADS=1 \n")
        script.write(" export OMP_NUM_THREADS=1 \n")
        script.write(" export OPENBLAS_NUM_THREADS=1 \n")

        self.script = script

    def add_batch(self,thescript,params):

        cmd = 'python {}'.format(thescript)

        for key,vals in params.items():
            cmd += ' --{} {}'.format(key,vals)

        self.script.write(cmd+'\n')



    def go_batch(self):
        """
        Method to close the batch script
        and to launch the batch
        """

        #self.script.write("EOF" + "\n")
        self.script.close()
        #os.system("sh "+scriptName)
        os.system("sbatch "+self.scriptName)
