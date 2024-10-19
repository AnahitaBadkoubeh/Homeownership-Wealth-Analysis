
import pandas as pd
from linearmodels.panel import PanelOLS

# Assuming you have a DataFrame named 'data' containing all the variables
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Table 5: Neighborhood Quality
def table_5(data):
    neighborhood_quality_measures = ['sfd_prcnt', 'white_prcnt', 'propoo', 'pvrty_prcnt']
    results_neighborhood_quality_df = pd.DataFrame()

    for measure in neighborhood_quality_measures:
        above_median_data = data[data[f'q2_{measure}'] == 2]
        below_median_data = data[data[f'q2_{measure}'] == 1]
        reg_above_median = PanelOLS.from_formula(f'wealth1 ~ 1 + EntityEffects + TimeEffects', above_median_data).fit(cov_type='clustered', cluster_entity=True)
        reg_below_median = PanelOLS.from_formula(f'wealth1 ~ 1 + EntityEffects + TimeEffects', below_median_data).fit(cov_type='clustered', cluster_entity=True)
        results_measure_df = pd.concat([reg_above_median.summary.tables[1], reg_below_median.summary.tables[1]], keys=[f'abv_{measure}', f'blw_{measure}'])
        results_neighborhood_quality_df = pd.concat([results_neighborhood_quality_df, results_measure_df])
    
    results_neighborhood_quality_df.to_csv('Table5.csv')
    print("Table 5 results exported to Table5.csv")

def main():
    file_path = 'rep_sample_psuedo.csv'
    data = load_dataset(file_path)
    table_5(data)

if __name__ == "__main__":
    main()
