import os
#get the current working directory
CWD = os.getcwd()

#Offensive Policy Environment storage path
PWD_OFFENSIVE = os.path.join(CWD, "ray_results", "OffensivePolicyEnv")
PWD_DEFENSIVE = os.path.join(CWD, "ray_results", "DefensivePolicyEnv")
