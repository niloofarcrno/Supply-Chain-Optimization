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
data2 = pd.read_excel('Data.xlsx', sheet_name='Demand', index_col=[0,1])

#define sets
i_set=[1,2,3]
j_set=[1,2,3]
l_set=[1,2,3,4,5,6,7,8]
k_set=[1,2,3,4,5,6,7,8]
r_set=[1,2]
t_set=[1,2,3,4,5]


all_t_set = [0] + t_set

for i, j in data.index:
    data_1=data.loc[(i, j), 'distance']
    
    
for k, t in data2.index:
    data_2=data2.loc[(k, t), 'demand']
    
    

model = pyo.ConcreteModel()


model.C = pyo.Var(i_set,all_t_set, bounds=(0,None)) #amount of cultivated 
C = model.C


model.I = pyo.Var(j_set,all_t_set,bounds=(0,None))
I = model.I

model.IB = pyo.Var(l_set,all_t_set,bounds=(0,None))
IB = model.IB

model.Q = pyo.Var(i_set,j_set,t_set, bounds=(0,None))#amount transported from supplier to collection center 
Q = model.Q

model.QBB= pyo.Var(j_set,l_set,r_set,t_set, bounds=(0,None))
QBB = model.QBB

model.QD = pyo.Var(l_set, k_set,r_set,t_set, bounds=(0,None))
QD = model.QD

model.P = pyo.Var(l_set,t_set, bounds=(0,None))
P = model.P


model.Z = pyo.Var(i_set,t_set, within=Binary)
Z = model.Z

model.Y= pyo.Var(j_set,t_set,  within=Binary)
Y = model.Y

model.X = pyo.Var(l_set,t_set, within=Binary)
X= model.X

model.CE = pyo.Var(j_set,all_t_set, bounds=(0,None))
CE= model.CE


model.WL = pyo.Var(l_set,all_t_set, bounds=(0,None))
WL= model.WL
#model.B = pyo.Var(i_set,t_set, within=Binary)
#B= model.B

#defining parameters
Conv = 0.9
Yield= 12
LC=3000
LW=5000
#defining parameters
model.Land = pyo.Param(i_set, initialize={1: 1000000, 2: 1000000, 3:1000000})  # Y
Land=model.Land

#Objective function (cost minimization)

cost=sum(200*model.Q[i,j,t]*data.distance[i,j] for i in data.index for j in data.index for i in i_set for j in j_set for t in t_set)+sum(100 * model.C[i,t] + 20 * model.P[l,t] + 20* model.P[l,t] for i in i_set for l in l_set for t in t_set)+sum(630*model.Y[j,t]+882000*X[l,t]+ 
500*model.Z[i,t] for i in i_set for l in l_set for j in j_set for t in t_set)+ sum(100*model.I[j,t]+50*model.IB[l,t] for l in l_set for j in j_set for t in t_set)
+sum(1000*CE[j,t] for j in j_set for t in t_set)

#constraints
#I_sum = sum([I[s,j,t] for s in s_set])

#model.balance1 = pyo.Constraint(expr = I_sum <= CE[j,t])

#land availablity 
model.land = pyo.Constraint(i_set,t_set, rule=lambda model, i, t: (
   C[i, t]<= Land[i]*Z[i,t]
))


model.capacity = pyo.Constraint(i_set, rule=lambda model, i: (
   C[i, 1]>= C[i,0] 
))

model.capacity2 = pyo.Constraint(i_set,t_set, rule=lambda model, i, t: (
   C[i, t]>= C[i, t-1] 
))


#Collection capacity 
model.capacitycoll = pyo.Constraint(j_set, rule=lambda model, j: (
   CE[j, 1]>= CE[j,0] 
))

model.capacitycoll2 = pyo.Constraint(j_set,t_set, rule=lambda model, j, t: (
   CE[j, t]>= CE[j, t-1] 
))

#Biorefinery capacity 
model.capacitybioref = pyo.Constraint(l_set, rule=lambda model, l: (
   WL[l, 1]>= WL[l,0] 
))

model.capacitybioref2 = pyo.Constraint(l_set,t_set, rule=lambda model, l, t: (
   WL[l, t]>= WL[l, t-1] 
))

#collection limitation
model.capacityCollLimit = pyo.Constraint(j_set,t_set, rule=lambda model, j, t: (
   CE[j, t]>= LC 
))

model.capacityCollLimit2 = pyo.Constraint(j_set,t_set, rule=lambda model, j, t: (
   CE[j, t]<= 100000*Y[j,t]
))


#biorefinery limitation
model.capacitybiorefLimit = pyo.Constraint(l_set,t_set, rule=lambda model, j, t: (
  WL[l, t]>= LW
))

model.capacitybiorefLimit2= pyo.Constraint(l_set,t_set, rule=lambda model, j, t: (
   WL[l, t]<= 100000*X[l,t]
))

#biomass yield
model.yiled = pyo.Constraint(i_set, t_set, rule=lambda model, i, t: (
    sum(Q[i,j,t] for j in j_set) ==  C[i,t]*Yield
))
#Collection capacity
model.harves= pyo.Constraint(j_set, t_set, rule=lambda model, j, t: (
    I[j,t] <= CE[j,t] 
))

#Biorefinery capacity
model.bioref= pyo.Constraint(l_set, t_set, rule=lambda model, l, t: (
    IB[l,t] <= WL[j,t] 
))


#biorefinery inventory
model.C1_initial = pyo.Constraint(l_set, rule=lambda model, l: (
     IB[l, 1] == IB[l, 0] + P[l, 1] - sum(QD[l, k, r, 1] for k in k_set for r in r_set)
))

# Constraints for the remaining time periods (t > 0)
model.C1 = pyo.Constraint(l_set, t_set, rule=lambda model, l, t: (
    IB[l, t] == IB[l, t-1] + P[l, t] - sum(QD[l, k, r, t] for k in k_set for r in r_set)
))

#Collection inventory
model.Collectinv = pyo.Constraint(j_set, rule=lambda model, j: (
     I[j, 1] == I[j, 0] + sum(Q[i,j,1]for i in i_set)-sum(QBB[j, l, r, 1] for l in l_set for r in r_set)
)) 

model.Collectinv2 = pyo.Constraint(j_set,t_set, rule=lambda model, j, t: (
     I[j, t] == I[j, t-1] + sum(Q[i,j,t] for i in i_set)-sum(QBB[j, l, r, t] for l in l_set for r in r_set)
)) 


#demand satisfaction
model.C3 = pyo.Constraint(k_set, t_set, rule=lambda model, k, t: (
    sum(QD[l,k,r,t] for r in r_set for l in l_set) >= data2.demand[k,t])
)

model.C5 = pyo.Constraint(t_set, l_set, rule=lambda model, t, l: (
    P[l,t] == sum(QBB[j,l,r,t]*Conv for j in j_set for r in r_set)
))

#Solving model                
model.obj = pyo.Objective(expr= cost, sense=minimize)

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

model.pprint()

for i in i_set:
    for t in t_set:
        C_value=model.C[i, t].value

    print("c",C_value)

obj=model.obj()
print(obj)
    
 
for i in i_set:
    for t in t_set:
        print(f"C[{i}, {t}] = {model.C[i, t].value}")

for l in l_set:
    for t in t_set:
        print(f"P[{l}, {t}] = {model.P[l, t].value}")

    for j in j_set:
        for t in t_set:
            print(f"I[{j}, {t}] = {model.I[j,t].value}")
            
for l in l_set:
    for t in t_set:
        print(f"X[{l}, {t}] = {model.X[l, t].value}")
    
for l in l_set:
    for k in k_set:
        for r in r_set:
            for t in t_set:
                print(f"QD[{l}, {k},{r}, {t}] = {model.QD[l, k, r, t].value}") 
           
# Create empty DataFrames to store the variables
data_C = {'Variable': [], 'Value': []}
data_P = {'Variable': [], 'Value': []}
data_I = {'Variable': [], 'Value': []}
data_X = {'Variable': [], 'Value': []}
data_QD = {'Variable': [], 'Value': []}

# Loop through your sets and collect the variable names and values
for i in i_set:
    for t in t_set:
        data_C['Variable'].append(f'C[{i}, {t}]')
        data_C['Value'].append(model.C[i, t].value)

for l in l_set:
    for t in t_set:
        data_P['Variable'].append(f'P[{l}, {t}]')
        data_P['Value'].append(model.P[l, t].value)

        data_I['Variable'].append(f'I[{j}, {t}]')
        data_I['Value'].append(model.I[j, t].value)

        data_X['Variable'].append(f'X[{l}, {t}]')
        data_X['Value'].append(model.X[l, t].value)

        for k in k_set:
            for r in r_set:
                for t in t_set:
                    data_QD['Variable'].append(f'QD[{l}, {k}, {r}, {t}]')
                    data_QD['Value'].append(model.QD[l, k, r, t].value)

# Create DataFrames from the collected data
df_C = pd.DataFrame(data_C)
df_P = pd.DataFrame(data_P)
df_I = pd.DataFrame(data_I)
df_X = pd.DataFrame(data_X)
df_QD = pd.DataFrame(data_QD)

# Save each DataFrame to a separate sheet in an Excel file
with pd.ExcelWriter('output.xlsx') as writer:
    df_C.to_excel(writer, sheet_name='C')
    df_P.to_excel(writer, sheet_name='P')
    df_I.to_excel(writer, sheet_name='I')
    df_X.to_excel(writer, sheet_name='X')
    df_QD.to_excel(writer, sheet_name='QD')            
            