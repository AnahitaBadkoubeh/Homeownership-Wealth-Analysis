
import os
import pandas as pd
from linearmodels.panel import PanelOLS
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Assuming you have a DataFrame named 'data' containing all the variables
# Load the dataset if needed
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Table 3: W.A. Mechanisms/Channels (Internal): Fragility
def table_3(data):
    dependent_variable = 'wealth1'
    highfrgl_hvstart_variable = 'highfrgl_hvstart'

    above_frgl = data[highfrgl_hvstart_variable] == 2  # high > median
    below_frgl = data[highfrgl_hvstart_variable] == 1  # low < median

    reg1 = PanelOLS.from_formula(f'{dependent_variable} ~ 1 + EntityEffects + TimeEffects', data[above_frgl]).fit(cov_type='clustered', cluster_entity=True)
    reg2 = PanelOLS.from_formula(f'{dependent_variable} ~ 1 + EntityEffects + TimeEffects', data[below_frgl]).fit(cov_type='clustered', cluster_entity=True)

    results_df = pd.concat([reg1.summary.tables[1], reg2.summary.tables[1]], keys=['abvsp50_frgl', 'blwsp50_frgl'])
    results_df.to_csv('Table3.csv')
    print("Table 3 results exported to Table3.csv")

# Table 4: Labor Supply
def table_4(data):
    data_ls = data[data['dsblty_indr'] == 0]
    reg_ls1 = PanelOLS.from_formula('total_wage_incm_amnt ~ 1 + EntityEffects + TimeEffects', data_ls).fit(cov_type='clustered', cluster_entity=True)
    reg_ls2 = PanelOLS.from_formula('netchg_wageincm ~ 1 + EntityEffects + TimeEffects', data_ls).fit(cov_type='clustered', cluster_entity=True)
    reg_ls3 = PanelOLS.from_formula('netchg_wagewrkn ~ 1 + EntityEffects + TimeEffects', data_ls).fit(cov_type='clustered', cluster_entity=True)
    results_ls_df = pd.concat([reg_ls1.summary.tables[1], reg_ls2.summary.tables[1], reg_ls3.summary.tables[1]], keys=['total_wage_incm_amnt', 'netchg_wageincm', 'netchg_wagewrkn'])
    results_ls_df.to_csv('Table4.csv')
    print("Table 4 results exported to Table4.csv")

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

# Table 7: Timing Test
def table_7(data):
    timing_measures = ['boom', 'bust', 'recovery']
    results_timing_df = pd.DataFrame()

    for measure in timing_measures:
        timing_data = data[data[measure] == 1]
        reg_timing = PanelOLS.from_formula(f'wealth1 ~ 1 + EntityEffects + TimeEffects', timing_data).fit(cov_type='clustered', cluster_entity=True)
        results_timing_df = pd.concat([results_timing_df, reg_timing.summary.tables[1]], keys=[f'timing_{measure}'])

    results_timing_df.to_csv('Table7.csv')
    print("Table 7 results exported to Table7.csv")

# Visualization: Differences in Household Wealth
def plot_wealth_differences(data):
    data_subset = data[(data['t'] > 11) & (data['t'] < 21)]
    model_v1 = smf.ols('wealth1 ~ nonwhite + t + tpre15 + tpost15 + nonwhite:t + nonwhite:tpre15 + nonwhite:tpost15', data=data_subset).fit()
    model_v2 = smf.ols('wealth1 ~ nonwhite + t + tpre15 + tpost15 + nonwhite:t + nonwhite:tpre15 + nonwhite:tpost15', data=data_subset).fit()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='t', y=model_v1.params['t'], hue='nonwhite', data=data_subset, marker='o', label='White Households', color='green')
    sns.lineplot(x='t', y=model_v2.params['t#1.nonwhite'], hue='nonwhite', data=data_subset, marker='o', label='Diff-in-Minority Households', color='blue')
    plt.axvline(x=4, color='red', linestyle='--', label='Vertical Line at x=4')
    plt.title('Differences in Household Wealth')
    plt.xlabel('Household Years, Relative to Start of Homeownership (t=0)')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.show()

# Main function to execute all tables and visualizations
def main():
    file_path = 'rep_sample_psuedo.csv'
    data = load_dataset(file_path)
    table_3(data)
    table_4(data)
    table_5(data)
    table_7(data)
    plot_wealth_differences(data)

if __name__ == "__main__":
    main()
