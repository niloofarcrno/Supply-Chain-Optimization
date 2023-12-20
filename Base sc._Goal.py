#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 01:04:03 2023

@author: niloofarakbarian
"""

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.environ import minimize
from pyomo.environ import Binary
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import os

os. chdir('/Users/niloofarakbarian/Library/CloudStorage/OneDrive-UBC/PhD-UT/Optimization')


#define each sheet
data = pd.read_excel('Data.xlsx', sheet_name='Path', index_col=[0,1])
data3 = pd.read_excel('Data.xlsx', sheet_name='Path2', index_col=[0,1])
data4 = pd.read_excel('Data.xlsx', sheet_name='Path3', index_col=[0,1])
data2 = pd.read_excel('Data.xlsx', sheet_name='Demand', index_col=[0,1])
data6 = pd.read_excel('Data.xlsx', sheet_name='Budget', index_col=[0])
#define sets
i_set=[1,2,3]
j_set=[1,2,3]
l_set=[1,2,3,4,5,6,7,8]
k_set=[1,2,3,4,5,6,7,8]
r_set=[1,2]# 1: truck, 2: train 
t_set=[1,2,3,4,5]


all_t_set = [0] + t_set

for i, j in data.index:
    data_1=data.loc[(i, j), 'distance']
    
    
for k, t in data2.index:
    data_2=data2.loc[(k, t), 'demand']
    
     
for t in data6.index:
    data_6=data6.loc[(t), 'Budget']
    
       

model = pyo.ConcreteModel()


model.C = pyo.Var(i_set, t_set, bounds=(0,None)) #amount of cultivated 
C = model.C


model.I = pyo.Var(j_set,all_t_set,bounds=(0,None))
I = model.I

model.IB = pyo.Var(l_set,all_t_set,bounds=(0,None))
IB = model.IB

model.Q = pyo.Var(i_set,j_set,t_set, bounds=(0,None))#amount transported from supplier to collection center 
Q = model.Q

model.QBC= pyo.Var(j_set,l_set,t_set, bounds=(0,None))
QBC = model.QBC

model.QBB = pyo.Var(l_set, k_set,r_set,t_set, bounds=(0,None))
QBB = model.QBB

model.P = pyo.Var(l_set,t_set, bounds=(0,None))
P = model.P


model.Z = pyo.Var(i_set, within=Binary)
Z = model.Z

model.Y= pyo.Var(j_set,  within=Binary)
Y = model.Y

model.X = pyo.Var(l_set, within=Binary)
X= model.X


model.SP = pyo.Var(j_set,t_set, bounds=(0,None))
SP= model.SP

model.HC= pyo.Var(i_set,t_set, bounds=(0,None))
HC= model.HC

model.deviation_cost = pyo.Var(within=pyo.NonNegativeReals, initialize=0)
deviation_cost = model.deviation_cost  # Add the variable to the model



model.deviation_emissions= pyo.Var(within=pyo.NonNegativeReals, initialize=0)
deviation_emissions =model.deviation_emissions 

#model.B = pyo.Var(i_set,t_set, within=Binary)
#B= model.B

#defining parameters
Conv = 0.34
Yield= 13
M=10000000
TC=0.05
FBC=0.8882
FCC=0.0063
VCB=7.5
CSP=0.015
Landa=0.135
CAPC=1500000
IBB=0.027
VWL=0.001
ISB=0.005
CAPB=2000000
HR=1.18
Teta=0.08


#emissions
EC=1600
EB=24452305
EH=17300
ECC=8500
EP=0.05
ETC=0.005
EHB=0.02
EHC=60
EPS=38500
ESQ=500

#defining parameters
model.Land = pyo.Param(i_set, initialize={1: 1000000, 2: 1000000, 3:700000})  # Y
Land=model.Land

model.FCL= pyo.Param(i_set, initialize={1: 1.45, 2: 1.52, 3:1.56})  # Y
FCL=model.FCL

model.TCR=pyo.Param(r_set, initialize={1:0.4, 2:0.2})
TCR=model.TCR

model.ETR=pyo.Param(r_set, initialize={1:0.004, 2:0.003})
ETR=model.ETR

model.VCC=pyo.Param(i_set, initialize={1:0.83, 2:0.9, 3: 0.87})
VCC=model.VCC

#Define goal achievement levels
target_cost = 500000  # Minimize total cost
target_emissions = 100000000 # Minimize GHG emissions
W1=0.6
W2= 0.4

#Objective function (cost minimization)

cost=(
sum(TC*model.Q[i,j,t]*data.distance[i,j] for i in i_set for j in j_set for t in t_set)
+sum(TC*model.QBC[j,l,t]*data3.distance[j,l] for j in j_set for l in l_set for r in r_set for t in t_set)
+sum(TCR[r]*model.QBB[l,k,r,t]*data4.distance[l,k] for l in l_set for k in k_set for r in r_set for t in t_set)
+sum(VCC[i]*C[i,t] for i in i_set for t in t_set)
+sum(VCB * P[l,t]  for l in l_set for t in t_set)
+sum(FCC*Y[j] for j in j_set)
+sum(FBC*X[l] for l in l_set)
+sum(FCL[i]*Z[i] for i in i_set)
+sum(ISB*I[j,t] for j in j_set for t in t_set)
+sum(IBB*IB[l,t] for l in l_set for t in t_set)
+sum(HR*HC[i,t] for i in i_set for t in t_set)
)

Emission=(
sum(ETC*model.Q[i,j,t]*data.distance[i,j] for i in i_set for j in j_set for t in t_set)
+sum(ETC*model.QBC[j,l,t]*data3.distance[j,l] for j in j_set for l in l_set for r in r_set for t in t_set)
+sum(ETR[r]*model.QBB[l,k,r,t]*data4.distance[l,k] for l in l_set for k in k_set for r in r_set for t in t_set)
+sum((ECC-ESQ)*C[i,t] for i in i_set for t in t_set)
+sum(EP* P[l,t]  for l in l_set for t in t_set)
+sum(EC*Y[j] for j in j_set)
+sum(EB*X[l] for l in l_set)
+sum(EHC*I[j,t] for j in j_set for t in t_set)
+sum(EHB*IB[l,t] for l in l_set for t in t_set)
+sum(EH*HC[i,t] for i in i_set for t in t_set)
)
#constraintsS
model.Cost_Goal_Constraint = pyo.Constraint(
    expr= (
    sum(TC*model.Q[i,j,t]*data.distance[i,j] for i in i_set for j in j_set for t in t_set)
    +sum(TC*model.QBC[j,l,t]*data3.distance[j,l] for j in j_set for l in l_set for r in r_set for t in t_set)
    +sum(TCR[r]*model.QBB[l,k,r,t]*data4.distance[l,k] for l in l_set for k in k_set for r in r_set for t in t_set)
    +sum(VCC[i]*C[i,t] for i in i_set for t in t_set)
    +sum(VCB * P[l,t]  for l in l_set for t in t_set)
    +sum(FCC*Y[j] for j in j_set)
    +sum(FBC*X[l] for l in l_set)
    +sum(FCL[i]*Z[i] for i in i_set)
    +sum(ISB*I[j,t] for j in j_set for t in t_set)
    +sum(IBB*IB[l,t] for l in l_set for t in t_set)
    +sum(HR*HC[i,t] for i in i_set for t in t_set))
     -deviation_cost == target_cost)

# GHG emissions constraint
model.add_component(
    "Emissions_Goal",
    pyo.Constraint(
        expr=(
        sum(ETC*model.Q[i,j,t]*data.distance[i,j] for i in i_set for j in j_set for t in t_set)
        +sum(ETC*model.QBC[j,l,t]*data3.distance[j,l] for j in j_set for l in l_set for r in r_set for t in t_set)
        +sum(ETR[r]*model.QBB[l,k,r,t]*data4.distance[l,k] for l in l_set for k in k_set for r in r_set for t in t_set)
        +sum((ECC-ESQ)*C[i,t] for i in i_set for t in t_set)
        +sum(EP* P[l,t]  for l in l_set for t in t_set)
        +sum(EC*Y[j] for j in j_set)
        +sum(EB*X[l] for l in l_set)
        +sum(EHC*I[j,t] for j in j_set for t in t_set)
        +sum(EHB*IB[l,t] for l in l_set for t in t_set)
        +sum(EH*HC[i,t] for i in i_set for t in t_set))
        - deviation_emissions == target_emissions)
    )

model.Cost = W1 * (deviation_cost) + W2 * deviation_emissions 


#land availablity /
model.landav = pyo.Constraint(i_set,t_set, rule=lambda model, i, t: (
   HC[i,t]== Yield*C[i,t]
))

#balance constraint
model.balance11 = pyo.Constraint(i_set, t_set, rule=lambda model, i, t: (
    sum(Q[i,j,t] for j in j_set)== HC[i,t]
))

#cultivation limitation
model.capacityCollLimit22 = pyo.Constraint(i_set,t_set, rule=lambda model, i,t: (
    C[i,t]<= Land[i]*Z[i] 
))


#Collection switchgrass
model.coll=pyo.Constraint(j_set, t_set, rule=lambda model, j,t: (
   sum(Q[i,j,t] for i in i_set)== SP[j,t]  
    ))
#Collection switchgrass capacity 

model.capcoll=pyo.Constraint(j_set, rule=lambda model, j: (
  SP[j,t] <=CAPC*Y[j]
  ))


#Collection switchgrass capacity 2

model.capcoll2=pyo.Constraint(j_set, t_set, rule=lambda model, j,t: (
  SP[j,t] >=Landa*CAPC*Y[j]
  ))  
  
#biorefinery capacity 

model.biorefcap=pyo.Constraint(l_set, t_set, rule=lambda model, l,t: (
  P[l,t] >=Teta*CAPB*X[l]
  ))  


#biorefinery capacity 2

model.biorefcap2=pyo.Constraint(l_set, t_set, rule=lambda model, l,t: (
  P[l,t] <= CAPB*X[l]
  ))  

#bioethanol production

model.product=pyo.Constraint(l_set, t_set, rule=lambda model, l,t: (
  P[l,t] == Conv*sum(QBC[j,l,t] for j in j_set)
  ))  

#biorefinery inventory
model.C1_initial = pyo.Constraint(l_set, rule=lambda model, l: (
     IB[l, 1] == IB[l, 0] + P[l, 1] - sum(QBB[l, k, r, 1] for k in k_set for r in r_set)
))

# Constraints for the remaining time periods (t > 0)
model.C1 = pyo.Constraint(l_set, t_set, rule=lambda model, l, t: (
    IB[l, t] == IB[l, t-1] + P[l, t]- sum(QBB[l, k, r, t] for k in k_set for r in r_set)
))

#Collection inventory
model.Collectinv = pyo.Constraint(j_set, rule=lambda model, j: (
     I[j, 1] == I[j, 0] + SP[j,1]-sum(QBC[j,l,1]for l in l_set)
)) 

model.Collectinv2 = pyo.Constraint(j_set,t_set, rule=lambda model, j, t: (
     I[j, t] == I[j, t-1] + + SP[j,t]-sum(QBC[j,l,t]for l in l_set)
)) 

#biorefinery balance

model.Binput = pyo.Constraint(l_set,t_set, rule=lambda model, l,t : (
    sum(QBB[l, k, r, t] for k in k_set for r in r_set) <=  P[l,t] )
)

#collection balance

model.collinput = pyo.Constraint(j_set,t_set, rule=lambda model, j,t : (
    sum(QBC[j,l, t] for l in l_set) <=  SP[j,t] )
)

#Inventory capacity
model.invswitch= pyo.Constraint(l_set, t_set, rule=lambda model, l, t: (
    I[j,t] <= CAPC*Y[j]
))


#Inventory capacity bioref
model.invbioref= pyo.Constraint(l_set, t_set, rule=lambda model, l, t: (
    IB[l,t] <= CAPB*X[l]
))


#demand satisfaction
model.C3 = pyo.Constraint(k_set, t_set, rule=lambda model, k, t: (
    sum(QBB[l,k,r,t] for r in r_set for l in l_set) == data2.demand[k,t])
)



# #installation 
# model.C9 = pyo.Constraint(l_set, k_set,r_set, t_set, rule=lambda model, l, k, r, t: (
#     QD[l,k,r,t]<=M*X[l,t])
# )

# model.C10 = pyo.Constraint(l_set, k_set,r_set, t_set, rule=lambda model, l, k, r, t: (
#    Q[i,j,t]<=M*Y[j,t])
# )

# def capacity_coll_limit_rule_positive(model, j, t):
#     return Y[j, t] >= CE[j, t]

# def capacity_coll_limit_rule_non_positive(model, j, t):
#     return Y[j, t] == 0

# model.capacityCollLimit2_positive = pyo.Constraint(j_set, t_set, rule=capacity_coll_limit_rule_positive)
# model.capacityCollLimit2_non_positive = pyo.Constraint(j_set, t_set, rule=capacity_coll_limit_rule_non_positive)

# model.capacity = pyo.Constraint(i_set, rule=lambda model, i: (
#     C[i, 1]>= C[i,0]
# ))

# model.capacity2 = pyo.Constraint(i_set,t_set, rule=lambda model, i, t: (
#     C[i, t]>= C[i, t-1] 
# ))
                            
model.obj = pyo.Objective(expr= model.Cost, sense=minimize)
opt = SolverFactory("gurobi", solver_io="python")
opt.solve(model)

obj= pyo.value(model.obj)  

# # Extract solution values
# solution_data = {'Variable_Name': [], 'Value': []}
# for var in model.component_objects(Var):
#     for index in var:
#         solution_data['Variable_Name'].append(var[index].name)
#         solution_data['Value'].append(value(var[index]))
# Create a DataFrame from the solution data
# solution_df = pd.DataFrame(solution_data)

obj=model.obj()
print(obj)
    
 
for i in i_set:
    for t in t_set:
        print(f"C[{i,t}) = {model.C[i,t].value}")

for l in l_set:
    for t in t_set:
        print(f"P[{l}, {t}] = {model.P[l, t].value}")

    for j in j_set:
        for t in t_set:
            print(f"I[{j}, {t}] = {model.I[j,t].value}")
            
for l in l_set:
        print(f"X[{l}] = {model.X[l].value}")
                
    

transp_components = {}
Harvesting_components={}
bioethanolproduction={}
investcost={}
Inventorycost={}

# Calculate the components and store them in the dictionary
for i in i_set:
    for j in j_set:
        for t in t_set:
            transp_components[(i, j, t)] = TC * model.Q[i, j, t].value * data.distance[i, j]

for j in j_set:
    for l in l_set:
        for r in r_set:
            for t in t_set:
                transp_components[(j, l, t)] = TC * model.QBC[j, l, t].value * data3.distance[j, l]

for l in l_set:
    for k in k_set:
        for r in r_set:
            for t in t_set:
                transp_components[(l, k, r, t)] = TCR[r] * model.QBB[l, k, r, t].value * data4.distance[l, k]

                
for i in i_set:
    for t in t_set:
        Harvesting_components[(i,t)]=VCC[i]*C[i,t].value
        
for i in i_set:
    for t in t_set:
        Harvesting_components[(i,t)]=HR*HC[i,t].value

        
for l in l_set:
    for t in t_set:
        bioethanolproduction[(l,t)]=VCB * P[l,t].value 

for i in i_set:
    investcost[(i)]=FCL[i]*Z[i].value

for l in l_set:
    investcost[(l)]=FBC*X[l].value

for j in j_set:
    investcost[(j)]=FCC*Y[j].value
    
    
for j in j_set:
   Inventorycost[(j,t)]=ISB*I[j,t].value
    
for l in l_set:
    Inventorycost[(l,t)]=IBB*IB[l,t].value
    
             
# Calculate the total Transp value by summing up the components
total_transp_value = sum(transp_components.values())
harvesting_costs=sum(Harvesting_components.values())
bioethanol_prod=sum(bioethanolproduction.values())
investcosttotal=sum(investcost.values())
totalinventory=sum(Inventorycost.values())
# Print the total Transp value
print("Total Transp Value:", total_transp_value)
print("harvesting costs:", harvesting_costs)
print("production costs:", bioethanol_prod)
print ("invest:", investcosttotal)
print ("inventory",totalinventory)
print ("total costs",totalinventory+investcosttotal+investcosttotal+ bioethanol_prod+harvesting_costs+total_transp_value)


transp_emissions = {}
harvesting_emissions = {}
investemissions = {}
production_emissions = {}
inventoryemissions = {}

# Calculate the components and store them in the dictionary
for i in i_set:
    for j in j_set:
        for t in t_set:
            transp_emissions[(i, j, t)] = ETC*model.Q[i,j,t].value*data.distance[i,j]

for j in j_set:
    for l in l_set:
        for r in r_set:
            for t in t_set:
                transp_emissions[(j, l, t)] = ETC*model.QBC[j,l,t].value*data3.distance[j,l]

for l in l_set:
    for k in k_set:
        for r in r_set:
            for t in t_set:
                transp_emissions[(l, k, r, t)] = ETR[r]*model.QBB[l,k,r,t].value*data4.distance[l,k]
                
for i in i_set:
    for t in t_set:
        harvesting_emissions[(i,t)]=EH*HC[i,t].value 
          
for i in i_set:
    for t in t_set:
        harvesting_emissions[(i,t)]=(ECC-ESQ)*C[i,t].value

              
for l in l_set:
    for t in t_set:
        production_emissions[(l,t)]=EP* P[l,t].value 

for l in l_set:
    investemissions[(l)]=EB*X[l].value

for j in j_set:
    investemissions[(j)]=EC*Y[j].value
    
    
for j in j_set:
   inventoryemissions[(j,t)]=EHC*I[j,t].value
    
for l in l_set:
    inventoryemissions[(l,t)]=EHB*IB[l,t].value
         

total_transp_value_emissions = sum(transp_emissions.values())
harvesting_emissions=sum(harvesting_emissions.values())
bioethanol_prod=sum(production_emissions.values())
investemission=sum(investemissions.values())
totalinventoryemissions=sum(inventoryemissions.values())
# Print the total Transp value
print("Total Transp emissions:", total_transp_value_emissions)
print("harvesting emissions:", harvesting_emissions)
print("production emissions:", bioethanol_prod)
print ("investemissions:", investemission)
print ("inventoryemissions",totalinventoryemissions)
print ("total emissions",totalinventoryemissions+investemission+ bioethanol_prod+harvesting_emissions+total_transp_value_emissions)


# Create a DataFrame with the total Transp value


# Save the DataFrame to an Excel file


# Create empty DataFrames to store the variables
data_C = {'Variable': [], 'Value': []}
data_P = {'Variable': [], 'Value': []}
data_I = {'Variable': [], 'Value': []}
data_IB = {'Variable': [], 'Value': []}
data_X = {'Variable': [], 'Value': []}
data_QBB = {'Variable': [], 'Value': []}
data_CE = {'Variable': [], 'Value': []}
data_Q = {'Variable': [], 'Value': []}
data_QBC = {'Variable': [], 'Value': []}
data_obj = {'Value': []}
data_transp = {'Value': []}
data_SP = {'Variable': [], 'Value': []}
data_Y= {'Variable': [], 'Value': []}
data_Z = {'Variable': [], 'Value': []}
data_HC={'Variable': [], 'Value': []}
# Loop through your sets and collect the variable names and values

data_obj['Value'].append(obj)
df=pd.DataFrame(data_obj)

data_transp['Value'].append(total_transp_value)


for i in i_set:
   
    data_Z['Variable'].append(f'Z{i}]')
    data_Z['Value'].append(model.Z[i].value)
 
    
        
for l in l_set:
    for t in t_set:
        data_P['Variable'].append(f'P[{l}, {t}]')
        data_P['Value'].append(model.P[l, t].value)

        data_I['Variable'].append(f'I[{j}, {t}]')
        data_I['Value'].append(model.I[j, t].value)

        data_X['Variable'].append(f'X[{l}]')
        data_X['Value'].append(model.X[l].value)
        
        data_IB['Variable'].append(f'IB[{l}, {t}]')
        data_IB['Value'].append(model.IB[l, t].value)

for l in l_set:
    for k in k_set:
        for r in r_set:
            for t in t_set:
                data_QBB['Variable'].append(f'QBB[{l}, {k}, {r}, {t}]')
                data_QBB['Value'].append(model.QBB[l, k, r, t].value)
        
           
for j in j_set:
    for l in l_set:
        for t in t_set:
                data_QBC['Variable'].append(f'QBC[{j}, {l}, {t}]')
                data_QBC['Value'].append(model.QBC[j, k, t].value)

for j in j_set:
    for t in t_set:
        data_SP['Variable'].append(f'SP[{j}, {t}]')
        data_SP['Value'].append(model.SP[j, t].value)
        
for i in i_set:
    for j in j_set:
        for t in t_set:
            data_Q['Variable'].append(f'Q {i}, [{j}, {t}]')
            data_Q['Value'].append(model.Q[i, j, t].value)        
            

for j in j_set:
    for t in t_set:
            data_Y['Variable'].append(f'Y{j}')
            data_Y['Value'].append(model.Y[j].value)     

for i in i_set:
    for t in t_set:                     
        data_HC['Variable'].append(f'HC{i,t}]')
        data_HC['Value'].append(model.HC[i,t].value)
        data_C['Variable'].append(f'C[{i},{t}]') 
        data_C['Value'].append(model.C[i,t].value)  
        
        
# Create DataFrames from the collected data
df_C = pd.DataFrame(data_C)
df_IB = pd.DataFrame(data_IB)
df_P = pd.DataFrame(data_P)
df_I = pd.DataFrame(data_I)
df_X = pd.DataFrame(data_X)
df_QBB = pd.DataFrame(data_QBB)
df_HC = pd.DataFrame(data_HC)
df_QBB = pd.DataFrame(data_QBB)
df_Q= pd.DataFrame(data_Q)
df_SP= pd.DataFrame(data_SP)
df_Y= pd.DataFrame(data_Y)
df_Z= pd.DataFrame(data_Z)
df_transp=pd.DataFrame(data_transp)


# Save each DataFrame to a separate sheet in an Excel file
with pd.ExcelWriter('base_goal.xlsx') as writer:
    df_C.to_excel(writer, sheet_name='C')
    df_P.to_excel(writer, sheet_name='P')
    df_I.to_excel(writer, sheet_name='I')
    df_IB.to_excel(writer, sheet_name='IB')
    df_X.to_excel(writer, sheet_name='X')
    df_QBB.to_excel(writer, sheet_name='QBB')            
    df_HC.to_excel(writer, sheet_name='HC')
    df_QBB.to_excel(writer, sheet_name='QBB')
    df_Q.to_excel(writer, sheet_name='Q')
    df_SP.to_excel(writer, sheet_name='SP')
    df_Y.to_excel(writer, sheet_name='Y')
    df_Z.to_excel(writer, sheet_name='Z')
    df.to_excel(writer, sheet_name='Obj')
    df_transp.to_excel(writer, sheet_name='transp')
