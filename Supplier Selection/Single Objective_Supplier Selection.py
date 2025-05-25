# Re-import necessary libraries after kernel reset
from pyomo.environ import *
from math import radians, cos, sin, sqrt, atan2
import pandas as pd

# -------------------------------
# Supplier Data
# -------------------------------
area_hectares = {
    'ARRATIA-NERVION': 14404.5967,
    'CANTABRICA ALAVESA': 8833.0665,
    'ENCARTACIONE': 10465.1737,
    'STRIBACIONESDEL GORBEA': 2778.8199,
    'GRANBILBAO': 3491.2616
}

# Define base yields per supplier (tonnes/hectare/year)
supplier_yields = {
    'ARRATIA-NERVION': 12.0,
    'CANTABRICA ALAVESA': 10.0,
    'ENCARTACIONE': 14.0,
    'STRIBACIONESDEL GORBEA': 9.0,
    'GRAN BILBAO': 8.0
}

# Define a Quality Index (QI) per supplier based on literature-informed assumptions (0 to 1 scale)
# Higher value = better quality (density, fewer defects, better log dimensions)
quality_index = {
    'ARRATIA-NERVION': 0.90,     # High-quality area, proximity to Murga
    'CANTABRICA ALAVESA': 0.75,  # Medium quality, inland
    'ENCARTACIONE': 0.85,        # Good conditions, coastal influence
    'ESTRIBACIONESDEL GORBEA': 0.70,  # Lower quality, mountain edge
    'GRAN BILBAO': 0.65          # Urban fringe, more variable quality
}
# Compute effective yields
effective_yields = {
    region: round(supplier_yields[region] * quality_index[region], 2)
    for region in supplier_yields
}

aac_rate = 0.25  # 25% annual allowable cut
aac_tonnes = {
    s: round(area_hectares[s] * supplier_yields[s] * aac_rate, 2)
    for s in area_hectares
}

supplier_coords = {
    'ARRATIA-NERVION': (43.07, -2.83),
    'CANTABRICA ALAVESA': (42.85, -2.68),
    'ENCARTACIONE': (43.22, -3.13),
    'ESTRIBACIONESDEL GORBEA': (43.07, -2.73),
    'GRANBILBAO': (43.26, -2.93)
}

facilities = {
    'Murga_Sawmill': {'lat': 43.07, 'lon': -3.05}
}

# -------------------------------
# Economic and Conversion Parameters
# -------------------------------
days_per_year = 240
price_lumber = 400
price_nano = 3000
daily_lumber_tonnes = 120
daily_nano_tonnes = 24
daily_log_input_tonnes = 208
lumber_conversion = daily_lumber_tonnes / daily_log_input_tonnes
nano_conversion = daily_nano_tonnes / daily_log_input_tonnes
annual_log_demand = daily_log_input_tonnes * days_per_year
fuel_cost_per_km_per_tonne = 0.044 * 1.51
harvest_cost_per_tonne = 50

# -------------------------------
# Haversine Distance Function
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

# -------------------------------
# Pyomo Optimization Model
# -------------------------------
model = ConcreteModel()

# Define Sets
model.S = Set(initialize=supplier_coords.keys())  # Suppliers
model.F = Set(initialize=facilities.keys())       # Facility

# Define Variables
model.x = Var(model.S, model.F, domain=NonNegativeReals)       # Log quantity from each supplier
model.y_lumber = Var(model.F, domain=NonNegativeReals)         # Output lumber
model.y_nano = Var(model.F, domain=NonNegativeReals)           # Output nano-cellulose

# Constraints

# Harvest cannot exceed allowable cut per supplier
def harvest_limit(m, s):
    return sum(m.x[s, f] for f in m.F) <= aac_tonnes[s]
model.harvest_limit = Constraint(model.S, rule=harvest_limit)

# Lumber production balance
def conversion_lumber(m):
    return sum(m.x[s, f] * quality_index[s] for s in m.S for f in m.F) * lumber_conversion == sum(m.y_lumber[f] for f in m.F)
model.lumber_balance = Constraint(rule=conversion_lumber)

def conversion_nano(m):
    return sum(m.x[s, f] * quality_index[s] for s in m.S for f in m.F) * nano_conversion == sum(m.y_nano[f] for f in m.F)
model.nano_balance = Constraint(rule=conversion_nano)

# Nano-cellulose production balance
def conversion_nano(m):
    return sum(m.x[s, f] for s in m.S for f in m.F) * nano_conversion == sum(m.y_nano[f] for f in m.F)
model.nano_balance = Constraint(rule=conversion_nano)

# Total log demand must be exactly met
def demand_constraint(m):
    return sum(m.x[s, f] for s in m.S for f in m.F) == annual_log_demand
model.demand_constraint = Constraint(rule=demand_constraint)

# Objective function: Maximize profit = revenue - harvesting - transportation
def objective_rule(m):
    revenue = sum(m.y_lumber[f] * price_lumber + m.y_nano[f] * price_nano for f in m.F)
    harvest_cost = sum(m.x[s, f] * harvest_cost_per_tonne for s in m.S for f in m.F)
    transport_cost = sum(
        m.x[s, f] * fuel_cost_per_km_per_tonne * haversine(
            supplier_coords[s][0], supplier_coords[s][1],
            facilities[f]['lat'], facilities[f]['lon']
        )
        for s in m.S for f in m.F
    )
    return revenue - harvest_cost - transport_cost

model.obj = Objective(rule=objective_rule, sense=maximize)
# Solve and Extract Results
solver = SolverFactory('glpk', executable='/opt/homebrew/bin/glpsol')
solver.solve(model, tee=True)


# Collect results
results = []
harvested_per_supplier = {s: 0.0 for s in model.S}
harvested_hectares = {}

for s in model.S:
    for f in model.F:
        val = model.x[s, f].value
        if val and val > 0:
            dist = haversine(supplier_coords[s][0], supplier_coords[s][1], facilities[f]['lat'], facilities[f]['lon'])
            transport_cost = val * fuel_cost_per_km_per_tonne * dist

            results.append({
                'Supplier': s,
                'Facility': f,
                'Tonnes Supplied': round(val, 2),
                'Distance (km)': round(dist, 2),
                'Transport Cost ($)': round(transport_cost, 2)
            })

            harvested_per_supplier[s] += val

# Convert to DataFrame for main results
df_results = pd.DataFrame(results)
print("\n--- Optimized Supplier Flows ---")
print(df_results)

# Print harvested tonnes and hectares per supplier
print("\n--- Total Harvested per Supplier ---")
for s, tonnes in harvested_per_supplier.items():
    hectares = tonnes / supplier_yields[s]
    harvested_hectares[s] = hectares
    print(f"{s}: {round(tonnes, 2)} tonnes → {round(hectares, 2)} hectares")
